# Video Processing Stream with WebRTC

A Python application that captures video from a USB camera, processes it through an extensible image processing pipeline, and streams it to a web browser using WebRTC.

## Features

- **Extensible Processing Pipeline**: Easily add custom image processing algorithms
- **Real-time WebRTC Streaming**: Low-latency video streaming to web browsers
- **Live Metrics Dashboard**: Display processing time and algorithm-specific metrics
- **Customizable Overlays**: Add timestamp, FPS, and custom overlays to the video
- **Modular Architecture**: Clean separation between algorithms, overlays, and streaming

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have a USB camera connected (or use the default webcam)

## Usage

### Basic Usage

Run the server:
```bash
python video_processor.py
```

Then open your browser and navigate to `http://localhost:8080`

### Adding Custom Algorithms

Create a new algorithm by subclassing `ProcessingAlgorithm`:

```python
class MyCustomAlgorithm(ProcessingAlgorithm):
    def __init__(self):
        super().__init__("My Algorithm")
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        start_time = time.time()
        
        # Your processing logic here
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # Update metrics
        process_time = (time.time() - start_time) * 1000
        self.metrics = {
            'process_time_ms': round(process_time, 2),
            'custom_metric': 42
        }
        
        return result
```

Add it to the pipeline in `CameraVideoTrack.__init__()`:
```python
self.pipeline.add_algorithm(MyCustomAlgorithm())
```

### Adding Custom Overlays

Create a new overlay by subclassing `OverlayItem`:

```python
class MyCustomOverlay(OverlayItem):
    def render(self, frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        cv2.putText(
            frame,
            "Custom Text",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        return frame
```

Add it to the pipeline in `CameraVideoTrack.__init__()`:
```python
self.pipeline.add_overlay(MyCustomOverlay())
```

## Architecture

### Components

- **ProcessingAlgorithm**: Base class for all image processing algorithms
- **OverlayItem**: Base class for video overlays
- **ImageProcessingPipeline**: Manages the processing pipeline and metrics collection
- **CameraVideoTrack**: WebRTC video track that integrates camera capture with processing
- **VideoStreamingServer**: HTTP/WebRTC server that handles streaming and metrics API

### Built-in Algorithms

1. **EdgeDetectionAlgorithm**: Canny edge detection with configurable thresholds
2. **GaussianBlurAlgorithm**: Gaussian blur with configurable kernel size

### Built-in Overlays

1. **TimestampOverlay**: Displays current timestamp
2. **FPSOverlay**: Displays frames per second
3. **MetricsOverlay**: Displays total processing time

## Configuration

### Camera Selection

Change the camera ID in `CameraVideoTrack`:
```python
self.camera = cv2.VideoCapture(0)  # 0 for default camera, 1, 2, etc. for others
```

### Server Port

Change the port in the main section:
```python
server = VideoStreamingServer(host='0.0.0.0', port=8080)
```

## Metrics API

Access real-time metrics via HTTP:
```
GET http://localhost:8080/metrics
```

Returns JSON with processing time, FPS, and algorithm-specific metrics.

## Troubleshooting

- **Camera not found**: Ensure your USB camera is connected and try different camera IDs
- **Port already in use**: Change the port number or kill the process using port 8080
- **Browser compatibility**: Use Chrome, Firefox, or Edge for best WebRTC support
