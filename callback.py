import rosnode
from std_msgs.msg import String
from khepera_communicator.msg import K4_controls, SensorReadings, Opt_Position
from geometry_msgs.msg import TransformStamped, PoseStamped
import tf_conversions


def callback(data, args):

    i = args

    theta = tf_conversions.transformations.euler_from_quaternion([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])[2]
    
    x[0][i] = data.pose.position.x
    x[1][i] = data.pose.position.y
    x[2][i] = theta

    
    
def central():

    sub.append(rospy.Subscriber('/vrpn_client_node/Hus111'  + '/pose', PoseStamped, callback, 0 ))
    sub.append(rospy.Subscriber('/vrpn_client_node/Hus111'  + '/pose', PoseStamped, callback, 1 ))
    sub.append(rospy.Subscriber('/vrpn_client_node/Hus111'  + '/pose', PoseStamped, callback, 2 ))
    sub.append(rospy.Subscriber('/vrpn_client_node/Hus111'  + '/pose', PoseStamped, callback, 3 ))
    rospy.spin()


if __name__ == '__main__':
	try:
		central()
	except rospy.ROSInterruptException:
		print(rospy.ROSInterruptException)
