# Music Genre Classification using Deep Learning

This repository contains the implementation of a **Convolutional Neural Network (CNN)** based model for **music genre classification** using **Mel Spectrograms** and **MFCCs**. The project demonstrates state-of-the-art performance on the GTZAN dataset with 94% validation accuracy.

## Project Files

├── Train_music_genre_classifier.ipynb # Training notebook  
├── Test_Music_Genre.ipynb # Testing/Evaluation notebook  
├── Trained_model.h5 # Saved model (HDF5 format)  
├── Trained_model.keras # Saved model (Keras format)  
└── training_hist.json # Training history data  

## Overview

Music genre classification is essential for music recommendation systems, playlist generation, and digital content organization. Traditional methods rely on handcrafted features, while this project uses deep learning—specifically CNNs—to automatically learn hierarchical representations from spectrograms and MFCCs.

Key Features:
- Feature extraction using Mel Spectrograms and MFCCs
- Custom CNN architecture with dropout and batch normalization
- Data augmentation (time stretching, pitch shifting, noise addition)
- High performance: 99% training accuracy, 94% validation accuracy
- Comprehensive evaluation with confusion matrix and metrics

## Model Architecture

The CNN processes 128x128 spectrogram images through multiple convolutional and pooling layers, followed by batch normalization, dropout, and fully connected layers for 10-genre classification.

<img width="626" height="327" alt="Image" src="https://github.com/user-attachments/assets/c77241dd-a1a0-4cca-82b0-cd152f6bb7e7" />

Model Flow Diagram

<img width="618" height="242" alt="Image" src="https://github.com/user-attachments/assets/a6fb13ce-7fb0-4133-b56a-e08b9181108d" />

Audio to Mel Spectrogram Representation

<img width="753" height="246" alt="Image" src="https://github.com/user-attachments/assets/97a6f192-a25f-4fe9-a92c-0fb13d79bbcd" />

CNN Architecture

## Results

Validation Performance:
- Overall Accuracy: 94%
- Precision (Macro Avg): 0.95
- Recall (Macro Avg): 0.94
- F1-Score (Macro Avg): 0.94

<img width="457" height="330" alt="Image" src="https://github.com/user-attachments/assets/e8ae62f2-feb2-412e-9f1e-df6ffee41ad3" />

Loss graph

<img width="489" height="335" alt="Image" src="https://github.com/user-attachments/assets/f7fbb1bb-0e9c-4787-93ac-76530be130d2" />

Accuracy graph

<img width="447" height="358" alt="Image" src="https://github.com/user-attachments/assets/b7cb7926-42d6-4b92-8b86-8fc7c616b41d" />

Confusion Matrix

## Usage

Training:
Open Train_music_genre_classifier.ipynb in Jupyter Notebook or Google Colab to load the dataset, extract features, train the CNN model, and save the trained model.

Testing/Evaluation:
Open Test_Music_Genre.ipynb to load the trained model, evaluate performance, generate metrics, and make predictions on new audio files.

## Dataset

The model is trained on the GTZAN dataset containing 1000 audio tracks (30 seconds each) across 10 genres:
Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock

## Technical Details

Model Specifications:
- Input: 128x128 spectrogram images
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Batch Size: 32
- Epochs: 30
- Regularization: Dropout (0.3–0.5), Batch Normalization

Feature Extraction:
- Sampling Rate: 22050 Hz
- STFT Window: 2048 samples
- Hop Length: 512 samples
- Mel Bands: 128
- MFCC Coefficients: 13

## Contributors

Bikash Naik – https://github.com/Bikashnaik07  
Subhankar Poddar  

Under the guidance of: Dr.Sanjit Kumar Dash

## Future Work

- Hybrid CNN-RNN architectures for temporal modeling
- Transfer learning with pre-trained audio models
- Real-time web/mobile deployment
- Multi-modal fusion with lyrics and metadata
