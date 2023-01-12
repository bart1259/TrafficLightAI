
port=3000

echo Creating tunnel on port ${port}
read -p "Enter node (dh-node2.hpc.msoe.edu): " node_name
read -p "Enter your ROSIE username (berisha): " username
echo Connecting to $node_name...
ssh -J ${username}@dh-mgmt4.hpc.msoe.edu -L 3000:127.0.0.1:3000 $node_name -t "echo \"Connection established. Press Ctrl+d then Ctrl+c to collapse tunnel\"; bash -l"
