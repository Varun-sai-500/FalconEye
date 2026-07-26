from core.following.follower import RoverFollower

follower_service = RoverFollower(
    stop_bbox_height=180,
    center_threshold=30,
    linear_speed=0.3,
    angular_speed=0.4,
)