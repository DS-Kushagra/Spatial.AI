# 🌐 Spatial.AI Web Interface

## Overview

The Spatial.AI Web Interface is a modern, interactive dashboard that provides a user-friendly way to interact with the GIS Workflow Generation System. Built with vanilla HTML, CSS, and JavaScript, it offers real-time query processing, Chain-of-Thought reasoning visualization, and dynamic map interactions.

## 🎯 Features

### 🧠 **Intelligent Query Interface**

- Natural language input with autocomplete
- Sample query buttons for common use cases
- Real-time query validation and suggestions
- Context-aware input assistance

### 🤖 **Chain-of-Thought Reasoning Display**

- Multi-perspective reasoning visualization
- Confidence scoring and consensus tracking
- Step-by-step reasoning breakdown
- Interactive reasoning exploration

### 🗺️ **Interactive Map Visualization**

- Dynamic Leaflet-based mapping
- Real-time result visualization
- Layer management and toggling
- Custom markers and overlays
- Fullscreen map mode

### ⚙️ **Workflow Management**

- Visual workflow step display
- Parameter inspection and editing
- Workflow export (JSON/YAML/Python)
- Code generation and download

### 📊 **Performance Monitoring**

- Real-time system metrics
- API health monitoring
- Processing time tracking
- Success rate analytics

### 🎨 **Modern UI/UX**

- Dark theme optimized for GIS work
- Responsive design for all devices
- Smooth animations and transitions
- Accessible interface design

## 🚀 Quick Start

### 1. Start the Web Interface

```bash
# Method 1: Using the main runner
python run.py web

# Method 2: Direct server start
cd web_interface
python server.py

# Method 3: Custom port
python server.py --port 3001
```

### 2. Start the API Backend

```bash
# In another terminal
python run.py api
```

### 3. Open Your Browser

The web interface will automatically open at `http://localhost:3000`

## 🏗️ Architecture

```
web_interface/
├── index.html          # Main HTML structure
├── styles.css          # Complete styling and themes
├── app.js             # Application logic and API integration
├── server.py          # Development web server
└── README.md          # This file
```

### Component Architecture

```
┌─────────────────────────────────────────────┐
│                Web Interface                │
├─────────────────┬───────────────────────────┤
│   Query Panel   │    Reasoning Panel        │
│                 │                           │
│ • Input Form    │ • Chain-of-Thought       │
│ • Sample Queries│ • Confidence Tracking    │
│ • Validation    │ • Multi-perspective View │
├─────────────────┼───────────────────────────┤
│   Map Panel     │    Workflow Panel        │
│                 │                           │
│ • Leaflet Map   │ • Step Visualization     │
│ • Layers        │ • Parameter Display      │
│ • Interactions  │ • Export Functions       │
├─────────────────┼───────────────────────────┤
│ Metrics Panel   │    Status Panel          │
│                 │                           │
│ • Performance   │ • System Health          │
│ • Statistics    │ • Component Status       │
└─────────────────┴───────────────────────────┘
```

## 🎛️ Interface Components

### Query Section

- **Input Area**: Large textarea with syntax highlighting
- **Sample Queries**: Pre-built query buttons for common scenarios
- **Options**: Checkboxes for reasoning and workflow inclusion
- **Submit Button**: Smart button with loading states

### Results Grid

- **Reasoning Panel**: Multi-perspective AI reasoning display
- **Map Panel**: Interactive Leaflet map with custom styling
- **Workflow Panel**: Step-by-step workflow visualization
- **Understanding Panel**: Query parsing and intent analysis
- **Metrics Panel**: Real-time performance indicators
- **Status Panel**: System component health monitoring

### Modals

- **Settings Modal**: API configuration and preferences
- **Code Viewer**: Generated code in multiple formats
- **Help Modal**: User guidance and documentation

## 🎨 Styling and Themes

### Dark Theme

The interface uses a carefully crafted dark theme optimized for GIS work:

```css
:root {
  --primary-bg: #0a0e1a; /* Deep dark background */
  --secondary-bg: #1a1f2e; /* Panel backgrounds */
  --panel-bg: #232937; /* Content panels */
  --accent-bg: #2a3441; /* Accent elements */
  --primary-blue: #3b82f6; /* Primary brand color */
  --text-primary: #f8fafc; /* Primary text */
  --text-secondary: #cbd5e1; /* Secondary text */
}
```

### Responsive Design

- Mobile-first approach with progressive enhancement
- Flexible grid system that adapts to screen size
- Touch-friendly interfaces for mobile devices
- Optimized performance on all devices

### Animations

- Smooth transitions for all interactions
- Loading animations with progress indicators
- Hover effects and micro-interactions
- Staggered animations for list items

## 🔧 API Integration

The web interface communicates with the Spatial.AI API through a comprehensive JavaScript client:

### API Client Features

```javascript
class SpatialAIDashboard {
    // Core API communication
    async callAPI(endpoint, data)

    // Query processing
    async analyzeQuery()

    // Status monitoring
    async checkApiStatus()
    async updateSystemStatus()

    // Result handling
    displayResults(result)
    updateMapWithResults(result)
}
```

### Endpoints Used

- `POST /query` - Main query processing
- `GET /health` - API health check
- `GET /status` - System status monitoring

## 🗺️ Map Integration

### Leaflet Configuration

```javascript
// Map initialization with dark theme
this.map = L.map("mapContainer", {
  zoomControl: true,
  scrollWheelZoom: true,
  doubleClickZoom: true,
  dragging: true,
}).setView([20.5937, 78.9629], 5);

// Dark theme tile layer
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "© OpenStreetMap contributors",
  maxZoom: 18,
}).addTo(this.map);
```

### Map Features

- **Interactive Markers**: Click for detailed information
- **Dynamic Layers**: Results displayed as overlays
- **Custom Styling**: Dark theme integration
- **Responsive Legend**: Auto-updating based on data
- **Fullscreen Mode**: Dedicated map view

## 📱 Mobile Responsiveness

### Breakpoints

```css
/* Tablet and below */
@media (max-width: 1200px) {
  .results-grid {
    grid-template-columns: 1fr 1fr;
  }
}

/* Mobile devices */
@media (max-width: 768px) {
  .results-grid {
    grid-template-columns: 1fr;
  }
}
```

### Mobile Optimizations

- Touch-friendly button sizes (44px minimum)
- Swipe gestures for panel navigation
- Optimized text sizes for readability
- Simplified layouts for small screens

## ⚡ Performance Optimizations

### JavaScript

- Lazy loading of heavy components
- Debounced API calls
- Efficient DOM manipulation
- Memory leak prevention

### CSS

- Hardware-accelerated animations
- Optimized selector specificity
- Minimal repaints and reflows
- Compressed and minified assets

### Network

- Request deduplication
- Response caching
- Parallel loading of resources
- Graceful error handling

## 🧪 Testing and Development

### Local Development

```bash
# Start development server with hot reload
cd web_interface
python server.py --port 3000

# Enable debug mode
export DEBUG=true
python server.py
```

### Browser Compatibility

- **Chrome**: Full support (recommended)
- **Firefox**: Full support
- **Safari**: Full support
- **Edge**: Full support
- **Mobile browsers**: Optimized support

### Debug Features

- Console logging for API calls
- Error boundary handling
- Performance timing logs
- State inspection tools

## 🔒 Security Considerations

### Content Security

- XSS prevention through input sanitization
- CSRF protection for state-changing operations
- Secure API communication
- Input validation and error handling

### Data Privacy

- No sensitive data stored in localStorage
- Session-based temporary storage only
- Secure API key handling
- User consent for data processing

## 📊 Analytics and Monitoring

### Built-in Metrics

```javascript
// Performance tracking
const metrics = {
  processingTime: "Query processing duration",
  confidenceScore: "AI confidence in results",
  workflowSteps: "Number of generated steps",
  complexityScore: "Workflow complexity rating",
};
```

### Health Monitoring

- Real-time API status
- Component health checks
- Error rate tracking
- Performance degradation alerts

## 🚀 Deployment

### Production Deployment

```bash
# Build optimized version
python build.py

# Start production server
python server.py --production --port 80
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim
COPY web_interface/ /app/
WORKDIR /app
EXPOSE 3000
CMD ["python", "server.py", "--port", "3000"]
```

### Environment Configuration

```bash
# Production settings
export PRODUCTION=true
export API_BASE_URL=https://api.spatial-ai.com
export PORT=3000
```

## 🤝 Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start the API server: `python run.py api`
4. Start the web interface: `python run.py web`
5. Make your changes and test thoroughly

### Code Style

- Use modern JavaScript (ES6+)
- Follow CSS BEM methodology
- Maintain responsive design principles
- Add comments for complex functionality

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request with detailed description

## 📄 License

This project is part of the Spatial.AI system and follows the same licensing terms.

---

**Built with ❤️ for the GIS and AI community**

For more information, visit the main [Spatial.AI documentation](../README.md).
