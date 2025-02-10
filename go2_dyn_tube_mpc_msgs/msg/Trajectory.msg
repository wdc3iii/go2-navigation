string MESSAGE_NAME="Trajectory"

std_msgs/Header header

float64[] z     # Trajectory States
float64[] v     # Trajectory Inputs
float64[] t     # Time associated with each node along the trajectory. Should have t[0] = header.stamp
int32 n         # dimension of state
int32 m         # dimension of input
int32 horizon   # length of horizon