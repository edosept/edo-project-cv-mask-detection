# Mask Detection

How AI Computer Vision helps enforce mask compliance against COVID-19 pandemic

![OpenCV detect mask](https://github.com/user-attachments/assets/c1520158-5e56-4a3d-a30c-a7d265da4a7b)

## Technical Approach
Uses OpenCV pre-trained face detector, combined with skin percentage analysis, to identify individuals who are wearing masks.

1. prepare your dataset use file Part 1 - Face detection, and detect your face.
2. normalize data image, img = (img - mean) / std.
3. calculate skin percent with analysis skin color histogram, and make sure to convert image to YCrCb color.

The technology enables automated monitoring in public spaces such as hospitals, transport hubs, and government buildings to ensure compliance with health protocols.

![Result](https://github.com/user-attachments/assets/033ffd54-df09-4080-a52e-c95569a335cb)
