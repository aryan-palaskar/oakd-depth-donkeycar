import logging
import time
from collections import deque
import cv2
import numpy as np
import depthai as dai

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CameraError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}'

class OakDCamera:
    def __init__(self, 
                 width, 
                 height, 
                 depth=3, 
                 isp_scale=None, 
                 framerate=30, 
                 enable_depth=False, 
                 enable_obstacle_dist=False, 
                 rgb_resolution="1080p",
                 rgb_apply_cropping=False,
                 rgb_sensor_crop_x=0.0,
                 rgb_sensor_crop_y=0.125,
                 rgb_video_size=(1280,600),
                 rgb_apply_manual_conf=False,
                 rgb_exposure_time = 2000,
                 rgb_sensor_iso = 1200,
                 rgb_wb_manual= 2800):

        self.on = False
        self.device = None
        self.rgb_resolution = rgb_resolution

        self.queue_xout = None
        self.queue_xout_depth = None
        self.queue_xout_spatial_data = None
        self.roi_distances = []

        self.frame_xout = None
        self.frame_xout_depth = None

        self.extended_disparity = False
        self.subpixel = False
        self.lr_check = True

        self.latencies = deque([], maxlen=20)
        self.enable_depth = enable_depth
        self.enable_obstacle_dist = enable_obstacle_dist

      
        self.pipeline = dai.Pipeline()
        self.pipeline.setXLinkChunkSize(0)

        if self.enable_depth:
            self.monoLeft = self.pipeline.create(dai.node.MonoCamera)
            self.monoRight = self.pipeline.create(dai.node.MonoCamera)
            self.depth = self.pipeline.create(dai.node.StereoDepth)
            self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            self.monoLeft.setCamera("left")
            self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            self.monoRight.setCamera("right")

        if self.enable_obstacle_dist:
            self.create_obstacle_dist_pipeline()

      
        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("xout")
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("xout_depth")

        
        if depth == 3:
            camera = self.pipeline.create(dai.node.ColorCamera)
            camera.setBoardSocket(dai.CameraBoardSocket.RGB)
            if self.enable_depth:
                self.depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
                self.depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
                self.depth.setLeftRightCheck(True)
                self.depth.setExtendedDisparity(False)
                self.depth.setSubpixel(False)

            if self.rgb_resolution == "800p":
                camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
            elif self.rgb_resolution == "1080p":
                camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            else:
                camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            camera.setInterleaved(False)
            camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

            if isp_scale:
                camera.setIspScale(isp_scale)
            if rgb_apply_cropping:
                camera.setSensorCrop(rgb_sensor_crop_x, rgb_sensor_crop_y)
                camera.setVideoSize(rgb_video_size)

            camera.setPreviewKeepAspectRatio(False)
            camera.setVideoSize(width, height)
            camera.setIspNumFramesPool(1)
            camera.setVideoNumFramesPool(1)
            camera.setPreviewNumFramesPool(1)

            if rgb_apply_manual_conf:
                camera.initialControl.setManualExposure(rgb_exposure_time, rgb_sensor_iso)
                camera.initialControl.setManualWhiteBalance(rgb_wb_manual)
            else:
                camera.initialControl.SceneMode(dai.CameraControl.SceneMode.SPORTS)
                camera.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
    
            camera.video.link(xout.input)
            if self.enable_depth:
                self.monoLeft.out.link(self.depth.left)
                self.monoRight.out.link(self.depth.right)
                self.depth.disparity.link(xout_depth.input)

        elif depth == 1:
          
            camera = self.pipeline.create(dai.node.MonoCamera)
            camera.setBoardSocket(dai.CameraBoardSocket.LEFT)
            camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

            manip = self.pipeline.create(dai.node.ImageManip)
            manip.setMaxOutputFrameSize(width * height)
            manip.initialConfig.setResize(width, height)
            manip.initialConfig.setFrameType(dai.RawImgFrame.Type.GRAY8)

            camera.out.link(manip.inputImage)
            manip.out.link(xout.input)
        else:
            raise ValueError("'depth' parameter must be either '3' (RGB) or '1' (GRAY)")


        camera.initialControl.setManualFocus(0)
        camera.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.FLUORESCENT)
        camera.setFps(framerate)

        try:
        
            logger.info('Starting OAK-D camera')
            self.device = dai.Device(self.pipeline)
            warming_time = time.time() + 5 

            if enable_depth:
                self.queue_xout = self.device.getOutputQueue("xout", maxSize=1, blocking=False)
                self.queue_xout_depth = self.device.getOutputQueue("xout_depth", maxSize=4, blocking=False)
                while (self.frame_xout is None or self.frame_xout_depth is None) and time.time() < warming_time:
                    logger.info("...warming RGB and depth cameras")
                    self.run()
                    time.sleep(0.2)
                if self.frame_xout is None:
                    raise CameraError("Unable to start OAK-D RGB and Depth camera.")
            elif enable_obstacle_dist:
                self.queue_xout = self.device.getOutputQueue("xout", maxSize=1, blocking=False)
                self.queue_xout_spatial_data = self.device.getOutputQueue("spatialData", maxSize=1, blocking=False)
            else:
                self.queue_xout = self.device.getOutputQueue("xout", maxSize=1, blocking=False)
                self.queue_xout_depth = None
                while self.frame_xout is None and time.time() < warming_time:
                    logger.info("...warming camera")
                    self.run()
                    time.sleep(0.2)
                if self.frame_xout is None:
                    raise CameraError("Unable to start OAK-D camera.")

            self.on = True
            logger.info("OAK-D camera ready.")
        except:
            self.shutdown()
            raise

    def create_obstacle_dist_pipeline(self):

        monoLeft = self.pipeline.create(dai.node.MonoCamera)
        monoRight = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)

        xoutSpatialData = self.pipeline.create(dai.node.XLinkOut)
        xinSpatialCalcConfig = self.pipeline.create(dai.node.XLinkIn)

        xoutSpatialData.setStreamName("spatialData")
        xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(True)
        spatialLocationCalculator.inputConfig.setWaitForMessage(False)

        for i in range(4):
            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 200
            config.depthThresholds.upperThreshold = 10000
            config.roi = dai.Rect(dai.Point2f(i*0.1+0.3, 0.35), dai.Point2f((i+1)*0.1+0.3, 0.43))
            spatialLocationCalculator.initialConfig.addROI(config)
            
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        stereo.depth.link(spatialLocationCalculator.inputDepth)
        spatialLocationCalculator.out.link(xoutSpatialData.input)
        xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    def run(self):

        if self.queue_xout is not None:
            data_xout = self.queue_xout.get()  
            image_data_xout = data_xout.getCvFrame()
            self.frame_xout = cv2.cvtColor(image_data_xout, cv2.COLOR_BGR2RGB)
            if logger.isEnabledFor(logging.DEBUG):
                self.latencies.append((dai.Clock.now() - data_xout.getTimestamp()).total_seconds() * 1000)
                if len(self.latencies) >= self.latencies.maxlen:
                    logger.debug('Image latency: {:.2f} ms, Avg: {:.2f} ms, Std: {:.2f}'.format(
                        self.latencies[-1], np.average(self.latencies), np.std(self.latencies)))
                    self.latencies.clear()

        if self.queue_xout_depth is not None:
            data_xout_depth = self.queue_xout_depth.get()
            disparity_frame = data_xout_depth.getFrame()  

            if self.depth is not None:
                max_disp = self.depth.initialConfig.getMaxDisparity()
            else:
                max_disp = 95

            disparity_norm = (disparity_frame * (255.0 / max_disp)).astype(np.uint16)
            self.frame_xout_depth = disparity_norm

    def run_threaded(self):

        if self.queue_xout is not None:
            data_xout = self.queue_xout.get()
            image_data_xout = data_xout.getCvFrame()
            self.frame_xout = cv2.cvtColor(image_data_xout, cv2.COLOR_BGR2RGB)

        if self.queue_xout_depth is not None:
            data_xout_depth = self.queue_xout_depth.get()
            disparity_frame = data_xout_depth.getFrame()

            if self.depth is not None:
                max_disp = self.depth.initialConfig.getMaxDisparity()
            else:
                max_disp = 95

            disparity_norm = (disparity_frame * (255.0 / max_disp)).astype(np.uint16)
            self.frame_xout_depth = disparity_norm

        if self.queue_xout_spatial_data is not None:
            xout_spatial_data = self.queue_xout_spatial_data.get().getSpatialLocations()
            self.roi_distances = []
            for depthData in xout_spatial_data:
                roi = depthData.config.roi
                coords = depthData.spatialCoordinates
                self.roi_distances.append(round(roi.topLeft().x, 2))
                self.roi_distances.append(round(roi.topLeft().y, 2))
                self.roi_distances.append(round(roi.bottomRight().x, 2))
                self.roi_distances.append(round(roi.bottomRight().y, 2))
                self.roi_distances.append(int(coords.x))
                self.roi_distances.append(int(coords.y))
                self.roi_distances.append(int(coords.z))
                
        if self.enable_depth:
            return self.frame_xout, self.frame_xout_depth
        elif self.enable_obstacle_dist:
            return self.frame_xout, np.array(self.roi_distances)
        else:
            return self.frame_xout

    def update(self):
        while self.on:
            self.run()
            time.sleep(0.001)

    def shutdown(self):
        self.on = False
        logger.info('Stopping OAK-D camera')
        time.sleep(0.5)
        if self.device is not None:
            self.device.close()
        self.device = None
        self.pipeline = None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture and save RGB and depth disparity images from the OAK-D camera")
    parser.add_argument("--width", type=int, default=1280, help="Width of the output image")
    parser.add_argument("--height", type=int, default=720, help="Height of the output image")
    parser.add_argument("--rgb_resolution", type=str, default="1080p", help="RGB camera resolution")
    parser.add_argument("--framerate", type=int, default=30, help="Camera framerate")
    args = parser.parse_args()

    try:
        camera = OakDCamera(width=args.width, height=args.height, depth=3, framerate=args.framerate, 
                            enable_depth=True, rgb_resolution=args.rgb_resolution)
        time.sleep(2)  

        rgb_frame, depth_frame = camera.run_threaded()

        if rgb_frame is not None:
            rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite("rgb_image.jpg", rgb_bgr)
            logger.info("Saved rgb_image.jpg")

        if depth_frame is not None:
            cv2.imwrite("depth_image.png", depth_frame)
            logger.info("Saved depth_image.png")

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'camera' in locals():
            camera.shutdown()
