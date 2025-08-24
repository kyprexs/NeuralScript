# NeuralScript Website

This is the official website for NeuralScript, a modern programming language for scientific computing and machine learning.

## 🌐 Live Website

The website is automatically deployed to GitHub Pages at: **https://kyprexs.github.io/NeuralScript/**

## 📁 Structure

```
website/
├── index.html              # Main homepage
├── assets/
│   ├── css/
│   │   └── style.css       # Modern dark theme styling
│   ├── js/
│   │   └── main.js         # Interactive features
│   └── images/             # Website assets
├── docs/
│   ├── getting-started.html # Installation and first steps
│   ├── language-reference.html # Language documentation
│   ├── api.html            # API reference
│   └── tutorials.html      # Learning materials
└── examples/               # Code examples and demos
```

## ✨ Features

- **🎨 Modern Design**: Dark theme with smooth animations
- **📱 Responsive**: Works perfectly on mobile and desktop
- **⚡ Interactive**: Tab switching, copy buttons, smooth scrolling
- **🎯 Performance**: Optimized for fast loading
- **♿ Accessible**: Keyboard navigation and screen reader support
- **🔍 SEO Optimized**: Meta tags and semantic HTML

## 🚀 Key Highlights

### Homepage Sections
- **Hero**: Striking introduction with live code example
- **Features**: 6 key feature cards with hover effects
- **Performance**: Real benchmark results in interactive tables
- **Examples**: Tabbed code examples (Neural Networks, Math, Physics)
- **Getting Started**: Step-by-step installation guide

### Technical Features
- **Syntax Highlighting**: Prism.js for beautiful code display
- **Particle Background**: Animated particle system in hero
- **Typewriter Effect**: Dynamic text animation
- **Copy to Clipboard**: One-click code copying
- **Mobile Menu**: Responsive hamburger navigation

## 🛠️ Local Development

To run the website locally:

1. **Simple HTTP Server** (Python):
   ```bash
   cd website
   python -m http.server 8000
   ```
   Visit: http://localhost:8000

2. **Live Server** (VS Code extension):
   - Install Live Server extension
   - Right-click `index.html` → "Open with Live Server"

## 📤 Deployment

The website automatically deploys to GitHub Pages when changes are pushed to the `main` branch. The deployment is configured via `.github/workflows/deploy-website.yml`.

### Manual Deployment
If you need to deploy manually:
1. Enable GitHub Pages in repository settings
2. Set source to "GitHub Actions"
3. Push changes to trigger automatic deployment

## 🎯 Performance Stats

The website showcases NeuralScript's impressive achievements:
- **340x GPU speedup** for ML operations
- **20,200+ lines** of production code
- **2.71x faster** than PyTorch
- **30.2% memory reduction** vs Python
- **3.74x JIT speedup** on average

## 🤝 Contributing

To contribute to the website:
1. Edit files in the `website/` directory
2. Test locally with a simple HTTP server
3. Submit a pull request with your changes

The website will automatically update when changes are merged to main.

---

*Making scientific computing as natural as mathematics itself.*
