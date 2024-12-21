# Solar Sunspot Detection using Deep Learning

A deep learning project that analyzes NASA Solar Dynamics Observatory (SDO) HMI Intensitygram data to detect and count sunspots, using labels from the Solar Influences Data Analysis Center (SIDC) of the Royal Observatory of Belgium.

## Project Overview
This project implements a deep neural network to automatically detect and count sunspots from solar imagery. It utilizes two approaches:

1. A custom-built neural network architecture
2. Transfer learning from existing pre-trained models

The model processes 512x512 pixel HMI Intensitygram images from NASA's SDO and predicts the number of sunspots present in the image.

## Data Sources

1. Images: HMI Intensitygram data from NASA Solar Dynamics Observatory (SDO)
2. Labels: Sunspot counts from the Solar Influences Data Analysis Center (SIDC), Royal Observatory of Belgium