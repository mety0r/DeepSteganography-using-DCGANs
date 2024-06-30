import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Function to build the generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(28 * 28 * 1, activation='tanh'))  # Output layer with tanh activation
    model.add(layers.Reshape((28, 28, 1)))  # Reshape to 28x28x1 image
    return model

# Function to build the discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation
    return model

# Function to train the GAN
def train_gan(generator, discriminator, gan, epochs=1000, batch_size=128, sample_interval=100):
    # Example: Implement training logic here
    pass

# Initialize Streamlit app
def main():
    st.title("GAN Image Generation with Streamlit")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = load_img(uploaded_file, target_size=(28, 28))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert the image to a numpy array and preprocess it
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Build the generator and discriminator
        generator = build_generator()
        discriminator = build_discriminator()

        # Compile the discriminator
        discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Compile the GAN
        discriminator.trainable = False
        gan_input = layers.Input(shape=(100,))
        generated_image = generator(gan_input)
        gan_output = discriminator(generated_image)
        gan = tf.keras.Model(gan_input, gan_output)
        gan.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the GAN (you may adjust epochs and batch_size)
        train_gan(generator, discriminator, gan, epochs=1000, batch_size=128, sample_interval=100)

        # Generate and display sample images
        st.subheader("Generated Images")
        # Replace with actual function to display generated images
        # sample_images(generator, 999)  # Change the epoch number as needed

if __name__ == '__main__':
    main()
 