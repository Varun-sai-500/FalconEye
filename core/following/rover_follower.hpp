#pragma once

struct Command {
    float linear;
    float angular;
    float error_x;
    float error_y;
    int   bbox_h;
    // 0=forward, 1=turning, 2=stop
    int   state;
};

class RoverFollower {
public:
    RoverFollower(
        int   frame_w          = 512,
        int   frame_h          = 512,
        int   stop_bbox_height = 180,
        float center_threshold = 30.0f,
        float linear_speed     = 0.3f,
        float angular_speed    = 0.4f
    )
        : frame_w_(frame_w),
          frame_h_(frame_h),
          stop_bbox_height_(stop_bbox_height),
          center_threshold_(center_threshold),
          linear_speed_(linear_speed),
          angular_speed_(angular_speed)
    {}

    Command compute(int x, int y, int w, int h) const {
        float cx      = x + w / 2.0f;
        float cy      = y + h / 2.0f;
        float error_x = cx - frame_w_ / 2.0f;
        float error_y = cy - frame_h_ / 2.0f;

        Command cmd{};
        cmd.error_x = error_x;
        cmd.error_y = error_y;
        cmd.bbox_h  = h;

        // too close — stop
        if (h >= stop_bbox_height_) {
            cmd.linear  = 0.0f;
            cmd.angular = 0.0f;
            cmd.state   = 2;   // stop
            return cmd;
        }

        // left/right steering
        if (error_x > center_threshold_)
            cmd.angular = -angular_speed_;   // turn right
        else if (error_x < -center_threshold_)
            cmd.angular =  angular_speed_;   // turn left
        else
            cmd.angular = 0.0f;             // centered

        // forward speed
        if (error_y < -center_threshold_)
            cmd.linear = linear_speed_;          // far — full speed
        else if (error_y > center_threshold_)
            cmd.linear = linear_speed_ * 0.4f;  // close-ish — creep
        else
            cmd.linear = linear_speed_;          // vertically centered

        cmd.state = (cmd.angular != 0.0f) ? 1 : 0;  // turning or forward
        return cmd;
    }

private:
    int   frame_w_;
    int   frame_h_;
    int   stop_bbox_height_;
    float center_threshold_;
    float linear_speed_;
    float angular_speed_;
};