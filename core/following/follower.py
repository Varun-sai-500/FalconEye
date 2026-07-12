class RoverFollower:
    def __init__(
        self,
        frame_w=512,
        frame_h=512,
        stop_bbox_height=180,   # px — approx 5ft away for a person
        center_threshold=30,    # px — dead zone, don't twitch for small errors
        linear_speed=0.3,       # m/s forward
        angular_speed=0.4,      # rad/s turn
    ):
        self.frame_w           = frame_w
        self.frame_h           = frame_h
        self.stop_bbox_height  = stop_bbox_height
        self.center_threshold  = center_threshold
        self.linear_speed      = linear_speed
        self.angular_speed     = angular_speed

    def compute(self, bbox: tuple) -> dict:
        """
        bbox: (x, y, w, h) in pixels
        returns: {linear, angular, state, error_x, error_y}
        """
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2

        # error from frame center (positive = right/down)
        error_x = cx - self.frame_w / 2
        error_y = cy - self.frame_h / 2

        # distance proxy — stop if object bbox is tall enough
        too_close = h >= self.stop_bbox_height

        if too_close:
            return {
                "linear":  0.0,
                "angular": 0.0,
                "state":   "stop",
                "error_x": round(error_x, 2),
                "error_y": round(error_y, 2),
                "bbox_h":  h,
            }

        # left/right — steer to center horizontally
        if error_x > self.center_threshold:
            angular = -self.angular_speed   # turn right
        elif error_x < -self.center_threshold:
            angular = self.angular_speed    # turn left
        else:
            angular = 0.0                   # centered, go straight

        # forward/back — drive toward object
        # error_y < 0 means object is high in frame = far = go forward
        # error_y > 0 means object is low in frame = close = slow down
        if error_y < -self.center_threshold:
            linear = self.linear_speed      # object far, move forward
        elif error_y > self.center_threshold:
            linear = self.linear_speed * 0.4  # object close-ish, creep
        else:
            linear = self.linear_speed      # vertically centered, full speed

        state = "turning" if angular != 0.0 else "forward"

        return {
            "linear":  round(linear, 3),
            "angular": round(angular, 3),
            "state":   state,
            "error_x": round(error_x, 2),
            "error_y": round(error_y, 2),
            "bbox_h":  h,
        }