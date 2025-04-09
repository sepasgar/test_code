import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Layer # Import the Layer class from tensorflow.keras.layers
from tensorflow.keras.layers import Input, MaxPooling2D,Conv2D, Conv2DTranspose, Flatten, Dense, Reshape # Import necessary layers
from tensorflow.keras.models import Model
from PIL import UnidentifiedImageError # Import UnidentifiedImageError


###########
# Set seeds for reproducibility
def set_seeds(seed=42):
    # Set Python seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set TensorFlow seed
    tf.random.set_seed(seed)
    
    # Set environment variables for even more determinism
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For older TensorFlow versions (< 2.7)
    # Note that this can impact performance
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        print("Op determinism not available in this TensorFlow version")
        
    print(f"Seeds set to {seed} for reproducibility")

# Call this function at the beginning of your script
set_seeds(42)  # You can choose any seed value
###########

# Data preprocessing
LATENT_DIM = 12 #16
IMG_WIDTH = 80  # original_size: 700
IMG_HEIGHT = 80  # original_size: 700
image_size = (IMG_WIDTH, IMG_HEIGHT)
CHANNELS = 1  # Changed from 3 to 1 for grayscale

def preprocess_image(image):
    image = tf.image.decode_png(image, channels=3)  # First decode with 3 channels
    image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

def load_image(image_path):
    image_file = tf.io.read_file(image_path)
    image = preprocess_image(image_file)
    return image

# Create TensorFlow Dataset
data_dir = r"C:\Users\user\Desktop\MSc_thesis\good" # Make sure this points to your image directory
image_paths = tf.data.Dataset.list_files(data_dir + "/*.png")
dataset = image_paths.map(load_image)
dataset = dataset.batch(32)  


#image = dataset.take(1)
#for batch in image:
#    plt.figure()
#    # Remove the extra dimension for display and specify cmap='gray'
#    plt.imshow(batch[0].numpy().squeeze(), cmap='gray')
#    plt.colorbar()  # Optional: shows the intensity scale
#    plt.title("Grayscale Image")
#    plt.axis('off')
#    plt.show()


#############


kernel_initializer = tf.keras.initializers.GlorotNormal(seed=42)


# Define the Generator model
def Generator():
    model = tf.keras.Sequential(name='Generator')

    model.add(tf.keras.layers.Dense(5 * 5 * 128, input_shape=(LATENT_DIM,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape((5, 5, 128)))

    model.add(tf.keras.layers.Conv2DTranspose(128, 3, kernel_initializer=kernel_initializer, strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(64, 3, kernel_initializer=kernel_initializer, strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

#    model.add(tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.Conv2DTranspose(64, 3, kernel_initializer=kernel_initializer, strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(32, 3, kernel_initializer=kernel_initializer, strides=2, padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, 3, kernel_initializer=kernel_initializer, strides=1, padding='same', activation='tanh'))


    return model


# Define the Discriminator model
def Discriminator():
    model = tf.keras.Sequential(name='Discriminator')
    model.add(tf.keras.layers.Conv2D(16, 3, kernel_initializer=kernel_initializer, strides=2, padding='same',
                                     input_shape=[IMG_WIDTH, IMG_HEIGHT, CHANNELS]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(32, 3, kernel_initializer=kernel_initializer, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer=kernel_initializer, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer=kernel_initializer, strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

generator = Generator()
discriminator = Discriminator()

#generator.summary()
#discriminator.summary()
#print(dataset)

#############

class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = LATENT_DIM


    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]


    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal([batch_size, self.latent_dim], seed=42)

        fake_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([fake_images, real_images], axis=0)

        labels = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors)
            predictions = self.discriminator(fake_images)
            g_loss = self.loss_fn(tf.ones((batch_size, 1)), predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics - this is the key part that's missing
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()
        }



class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, dataset, num_examples=3, latent_dim=LATENT_DIM):
        self.dataset = dataset
        self.num_examples = num_examples
        self.latent_dim = latent_dim
        self.seed = tf.random.normal([num_examples, latent_dim], seed=42)
        
        # Get some real images for comparison
        for batch in dataset.take(1):
            self.real_images = batch[:num_examples]
            break

    def on_epoch_end(self, epoch, logs=None):
        # Generate images from the same seed every time
        generated_images = self.model.generator(self.seed, training=False)
        
        # Plot the images: real vs generated
        plt.figure(figsize=(12, 6))
        
        # Plot real images
        for i in range(self.num_examples):
            plt.subplot(2, self.num_examples, i+1)
            img = self.real_images[i].numpy()
            # Rescale from [-1, 1] to [0, 1] if needed
            if img.min() < 0:
                img = (img + 1) / 2
            if img.shape[-1] == 1:  # If grayscale
                plt.imshow(img.squeeze(), cmap='gray')
            else:
                plt.imshow(img)
            plt.title("Real")
            plt.axis('off')
        
        # Plot generated images
        for i in range(self.num_examples):
            plt.subplot(2, self.num_examples, i+1+self.num_examples)
            img = generated_images[i].numpy()
            # Rescale from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            if img.shape[-1] == 1:  # If grayscale
                plt.imshow(img.squeeze(), cmap='gray')
            else:
                plt.imshow(img)
            plt.title("Generated")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"C:\\Users\\user\\Desktop\\MSc_thesis\\GANmonitor\\gan_comparison_epoch_{epoch+1}.png")
        plt.close()
        
        print(f"\nSaved comparison images at epoch {epoch+1}")



class GANCombinedEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=25, min_delta=0.0001, monitor='g_loss', mode='min', verbose=1):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as improvement
            monitor: Metric to monitor ('g_loss', 'd_loss', or 'ratio')
            mode: 'min' if lower value is better, 'max' if higher value is better
            verbose: Verbosity mode (0 or 1)
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        # Set initial best value depending on mode
        if self.mode == 'min':
            self.best = float('inf')
            self.monitor_op = lambda current, best: current < best - min_delta
        else:
            self.best = float('-inf')
            self.monitor_op = lambda current, best: current > best + min_delta
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        if self.mode == 'min':
            self.best = float('inf')
        else:
            self.best = float('-inf')
        
    def on_epoch_end(self, epoch, logs=None):
        d_loss = logs.get('d_loss')
        g_loss = logs.get('g_loss')
        
        if d_loss is None or g_loss is None:
            if self.verbose > 0:
                print("Warning: d_loss or g_loss not found in logs")
            return
            
        # Calculate the metric to monitor
        if self.monitor == 'd_loss':
            current = d_loss
        elif self.monitor == 'g_loss':
            current = g_loss
        elif self.monitor == 'ratio':
            if g_loss < 1e-7:  # Prevent division by zero
                current = 1000 if self.mode == 'min' else 0
            else:
                current = d_loss / g_loss
        else:
            raise ValueError(f"Unknown monitor: {self.monitor}")
            
        if self.verbose > 0:
            print(f"\nEpoch {epoch}: d_loss={d_loss:.4f}, g_loss={g_loss:.4f}, " + 
                  f"ratio={d_loss/g_loss:.4f}, {self.monitor}={current:.4f}")
        
        # Check if improved
        if self.monitor_op(current, self.best):
            if self.verbose > 0:
                print(f"{self.monitor} improved from {self.best:.4f} to {current:.4f}")
            self.best = current
            self.wait = 0
            self.best_weights = self._get_weights()
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"{self.monitor} did not improve. Wait: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self._set_weights(self.best_weights)
                if self.verbose > 0:
                    print(f'\nEarly stopping: Best {self.monitor} {self.best:.4f} achieved at epoch {epoch - self.wait}')
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'\nTraining stopped early at epoch {self.stopped_epoch + 1}')
    
    def _get_weights(self):
        """Safely get model weights"""
        try:
            return {
                'generator': self.model.generator.get_weights(),
                'discriminator': self.model.discriminator.get_weights()
            }
        except AttributeError:
            # Alternative approach if model structure is different
            return self.model.get_weights()
    
    def _set_weights(self, weights):
        """Safely set model weights"""
        try:
            self.model.generator.set_weights(weights['generator'])
            self.model.discriminator.set_weights(weights['discriminator'])
        except (AttributeError, TypeError):
            # Alternative approach if model structure is different
            self.model.set_weights(weights)

###########

gan = GAN(generator, discriminator, LATENT_DIM)

# Call compile() on the GAN instance
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00008),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.00012),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)
# Pass the callback to the fit method

#gan.summary()
monitor = GANMonitor(dataset)

# For your case, monitor g_loss since it's steadily decreasing
early_stopping = GANCombinedEarlyStopping(
    patience=25,
    min_delta=0.0001,  # Minimum improvement in g_loss
    monitor='g_loss',  # Monitor generator loss
    mode='min',  # Lower values are better
    verbose=1
)

#gan.fit(dataset, epochs=20, callbacks=[monitor])
gan.fit(dataset, epochs=5000, callbacks=[monitor])



# Save the generator weights
gan.generator.save_weights('generator_weights_5000_epochs.h5')


#######


# Generate images
num_examples_to_generate = 9
random_latent_vectors = tf.random.normal(shape=(num_examples_to_generate, LATENT_DIM))
generated_images = generator(random_latent_vectors)

# Denormalize images for display
generated_images = (generated_images * 127.5) + 127.5
generated_images = generated_images.numpy().astype(np.uint8)

# Plot the generated images
plt.figure(figsize=(10, 10))
for i in range(num_examples_to_generate):
    plt.subplot(3, 3, i + 1)
    plt.imshow(generated_images[i], cmap='gray')
    plt.axis('off')
plt.show()
