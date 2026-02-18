import cv2
import numpy as np
import asyncio
import json
import time
import os
import pathlib
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from av import VideoFrame


class ProcessingAlgorithm(ABC):
    """Base class for image processing algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.metrics = {}
    
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process the frame and return the result"""
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return current metrics for this algorithm"""
        return self.metrics.copy()


class EdgeDetectionAlgorithm(ProcessingAlgorithm):
    """Canny edge detection algorithm"""
    
    def __init__(self, threshold1=100, threshold2=200):
        super().__init__("Edge Detection")
        self.threshold1 = threshold1
        self.threshold2 = threshold2
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        start_time = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.threshold1, self.threshold2)
        
        # Convert back to BGR for display
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        process_time = (time.time() - start_time) * 1000
        
        # Calculate edge density as a metric
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = (edge_pixels / total_pixels) * 100
        
        self.metrics = {
            'process_time_ms': round(process_time, 2),
            'edge_density_percent': round(edge_density, 2),
            'threshold1': self.threshold1,
            'threshold2': self.threshold2
        }
        
        return result


class GaussianBlurAlgorithm(ProcessingAlgorithm):
    """Gaussian blur algorithm"""
    
    def __init__(self, kernel_size=15):
        super().__init__("Gaussian Blur")
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        start_time = time.time()
        
        result = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), 0)
        
        process_time = (time.time() - start_time) * 1000
        
        self.metrics = {
            'process_time_ms': round(process_time, 2),
            'kernel_size': self.kernel_size
        }
        
        return result


class OverlayItem(ABC):
    """Base class for overlay items"""
    
    @abstractmethod
    def render(self, frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        """Render the overlay item on the frame"""
        pass


class TimestampOverlay(OverlayItem):
    """Overlay that displays the current timestamp"""
    
    def render(self, frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        cv2.putText(
            frame,
            timestamp,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        return frame


class FPSOverlay(OverlayItem):
    """Overlay that displays FPS"""
    
    def render(self, frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        fps = data.get('fps', 0)
        
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        return frame


class MetricsOverlay(OverlayItem):
    """Overlay that displays processing metrics"""
    
    def render(self, frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        total_time = data.get('total_processing_time_ms', 0)
        
        cv2.putText(
            frame,
            f"Processing: {total_time:.2f}ms",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        return frame


class ImageProcessingPipeline:
    """Manages the image processing pipeline"""
    
    def __init__(self):
        self.algorithms: List[ProcessingAlgorithm] = []
        self.overlays: List[OverlayItem] = []
        self.total_processing_time = 0
        self.fps = 0
        self.last_time = time.time()
        self.frame_count = 0
    
    def add_algorithm(self, algorithm: ProcessingAlgorithm):
        """Add a processing algorithm to the pipeline"""
        self.algorithms.append(algorithm)
    
    def add_overlay(self, overlay: OverlayItem):
        """Add an overlay item to the pipeline"""
        self.overlays.append(overlay)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a frame through all algorithms and add overlays"""
        start_time = time.time()
        
        result = frame.copy()
        
        # Apply all enabled algorithms
        for algorithm in self.algorithms:
            if algorithm.enabled:
                result = algorithm.process(result)
        
        self.total_processing_time = (time.time() - start_time) * 1000
        
        # Calculate FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        # Prepare overlay data
        overlay_data = {
            'total_processing_time_ms': self.total_processing_time,
            'fps': self.fps
        }
        
        # Apply all overlays
        for overlay in self.overlays:
            result = overlay.render(result, overlay_data)
        
        # Collect metrics from all algorithms
        metrics = {
            'total_processing_time_ms': round(self.total_processing_time, 2),
            'fps': round(self.fps, 2),
            'algorithms': {}
        }
        
        for algorithm in self.algorithms:
            if algorithm.enabled:
                metrics['algorithms'][algorithm.name] = algorithm.get_metrics()
        
        return result, metrics


class CameraVideoTrack(VideoStreamTrack):
    """WebRTC video track that streams processed camera frames"""
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera = cv2.VideoCapture(camera_id)
        self.pipeline = ImageProcessingPipeline()
        
        # Add default algorithms
        self.pipeline.add_algorithm(EdgeDetectionAlgorithm())
        
        # Add default overlays
        self.pipeline.add_overlay(TimestampOverlay())
        self.pipeline.add_overlay(FPSOverlay())
        self.pipeline.add_overlay(MetricsOverlay())
        
        self.latest_metrics = {}
    
    async def recv(self):
        """Receive the next video frame"""
        pts, time_base = await self.next_timestamp()
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        # Process the frame
        processed_frame, metrics = self.pipeline.process_frame(frame)
        self.latest_metrics = metrics
        
        # Convert to VideoFrame
        new_frame = VideoFrame.from_ndarray(processed_frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        
        return new_frame
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the latest metrics"""
        return self.latest_metrics


class VideoStreamingServer:
    """WebRTC video streaming server"""
    
    def __init__(self, host='0.0.0.0', port=8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.pcs = set()
        self.relay = MediaRelay()
        self.camera_track = None
        
        # Ensure templates directory exists
        root_dir = pathlib.Path(__file__).parent.absolute()
        template_dir = os.path.join(root_dir, 'templates')
        os.makedirs(template_dir, exist_ok=True)
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/', self.index)
        self.app.router.add_post('/offer', self.offer)
        self.app.router.add_get('/metrics', self.metrics)
    
    async def index(self, request):
        """Serve the main HTML page"""
        # Get the path to the templates directory
        root_dir = pathlib.Path(__file__).parent.absolute()
        template_path = os.path.join(root_dir, 'templates', 'index.html')
        
        # Read the HTML file
        try:
            with open(template_path, 'r') as file:
                html = file.read()
            return web.Response(text=html, content_type='text/html')
        except FileNotFoundError:
            return web.Response(text="HTML template not found", status=500)
    
    async def offer(self, request):
        """Handle WebRTC offer"""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
        
        pc = RTCPeerConnection()
        self.pcs.add(pc)
        
        @pc.on('connectionstatechange')
        async def on_connectionstatechange():
            if pc.connectionState == 'failed' or pc.connectionState == 'closed':
                await pc.close()
                self.pcs.discard(pc)
        
        if self.camera_track is None:
            self.camera_track = CameraVideoTrack()
        
        await pc.setRemoteDescription(offer)
        
        pc.addTrack(self.relay.subscribe(self.camera_track))
        
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return web.Response(
            content_type='application/json',
            text=json.dumps({
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type
            })
        )
    
    async def metrics(self, request):
        """Return current metrics as JSON"""
        if self.camera_track:
            metrics = self.camera_track.get_metrics()
        else:
            metrics = {'error': 'No camera track available'}
        
        return web.Response(
            content_type='application/json',
            text=json.dumps(metrics)
        )
    
    async def on_shutdown(self, app):
        """Cleanup on shutdown"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()
        
        if self.camera_track:
            self.camera_track.camera.release()
    
    def run(self):
        """Start the server"""
        self.app.on_shutdown.append(self.on_shutdown)
        print(f"Starting video streaming server on http://{self.host}:{self.port}")
        print(f"Open your browser and navigate to http://localhost:{self.port}")
        web.run_app(self.app, host=self.host, port=self.port)


if __name__ == '__main__':
    server = VideoStreamingServer(host='0.0.0.0', port=8080)
    server.run()
