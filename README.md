# Neural Style Transfer Web App 🎨

Transform your photos into artistic masterpieces using the power of AI! This web application implements Neural Style Transfer using PyTorch, allowing you to apply the artistic style of one image to the content of another.

## Demo Video
<div style="position: relative; padding-bottom: 44.0625%; height: 0;"><iframe src="https://www.loom.com/embed/850abd53da9f43e2b18d2b6f784920fe?sid=1de48295-3eb4-432b-8815-260c427e4000" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

## Features ✨

- 🖼️ Upload content and style images through an intuitive interface
- 🔄 Real-time image preview before processing
- 🎯 Fast and efficient style transfer using VGG19
- 📱 Responsive design that works on both desktop and mobile
- ⬇️ Download functionality for the generated artwork
- 🎨 Beautiful, modern UI built with TailwindCSS

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/style_transfer.git
cd style_transfer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage 💻

1. Start the FastAPI server:
```bash
uvicorn main:app --reload
```

2. Open your browser and navigate to `http://localhost:8000`

3. Upload your content image (the base image you want to transform)

4. Upload your style image (the image whose artistic style you want to apply)

5. Click "Transform Image" and wait for the magic to happen!

## How It Works 🔍

The application uses a deep learning technique called Neural Style Transfer, which was introduced by Gatys et al. The process involves:

1. Using a pre-trained VGG19 network to extract content and style features
2. Optimizing an output image to match the content features of the content image and style features of the style image
3. Using Gram matrices to capture style information
4. Iteratively updating the output image to minimize both content and style losses

## Technical Details 🛠️

- **Backend**: FastAPI + Python
- **Deep Learning**: PyTorch + VGG19
- **Frontend**: HTML + TailwindCSS + AlpineJS
- **Image Processing**: PIL (Python Imaging Library)

## Learn More About Style Transfer 📚

Here are some excellent resources to learn more about Neural Style Transfer:

1. [Original Paper: A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
2. [PyTorch Tutorial on Neural Style Transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
3. [Towards Data Science Article on Style Transfer](https://towardsdatascience.com/neural-style-transfer-tutorial-part-1-f5cd3315fa7f)
4. [Stanford's CS231n Notes on Style Transfer](http://cs231n.github.io/neural-style-transfer/)

## Tips for Best Results 💡

1. **Content Images**:
   - Use clear, well-lit images
   - Avoid images with too many small details
   - Resolution of 256x256 or higher recommended

2. **Style Images**:
   - Choose images with distinct artistic styles
   - Paintings, drawings, or artistic photographs work best
   - Strong patterns and textures give interesting results

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Implementation inspired by the original Neural Style Transfer paper by Gatys et al.
- VGG19 model architecture from the Visual Geometry Group at Oxford
- UI design inspired by modern web design principles 