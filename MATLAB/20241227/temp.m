folderPath = '20241227_test3.db3';
bagReader = ros2bagreader(folderPath);
baginfo   = ros2("bag","info",folderPath);
msgs      = readMessages(bagReader);

cam_pos = msgs{1,1}.data(1:3);
cam_quat = msgs{1,1}.data(4:7);
target_pos = msgs{1,1}.data(8:10);
target_quat = msgs{1,1}.data(11:14);

ot = target_pos; 
oc = cam_pos;

at = quat2rotm(target_quat');
ac = quat2rotm(cam_quat');

scale = 0.1;

figure; hold on; axis equal; grid on; 
ylabel('Y'); xlabel('X'); zlabel('Z')
quiver3(ot(1), ot(2), ot(3), scale*at(1,1),scale*at(2,1),scale*at(3,1), 'r');
quiver3(ot(1), ot(2), ot(3), scale*at(1,2), scale*at(2,2), scale*at(3,2), 'g');
quiver3(ot(1), ot(2), ot(3), scale*at(1,3), scale*at(2,3), scale*at(3,3), 'b');

quiver3(oc(1), oc(2), oc(3), scale*ac(1,1), scale*ac(2,1), scale*ac(3,1), 'c');
quiver3(oc(1), oc(2), oc(3), scale*ac(1,2), scale*ac(2,2), scale*ac(3,2), 'p');
quiver3(oc(1), oc(2), oc(3), scale*ac(1,3), scale*ac(2,3), scale*ac(3,3), 'k');
