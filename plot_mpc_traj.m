fid = fopen('debugging_dtmpc.csv', 'r');
if fid == -1
    error('Cannot open file: %s', filename);
end
% Read and process each line
while ~feof(fid)
    line = fgetl(fid);
    if ischar(line)
        parts = strsplit(line, ',');  % Split by comma
        varName = strtrim(parts{1});  % First element as variable name
        values = str2double(parts(2:end));  % Convert remaining to numbers
        
        % Assign variable dynamically
        assignin('base', varName, values);
    end
end
fclose(fid);

z = reshape(z, 3, 21);
z_ref = reshape(zref, 3, 21);

v = reshape(v, 3, 20);
v_ref = reshape(vref, 3, 20);

figure(1)
clf
subplot(1,2,1)
hold on
plot(z_ref', '--')
set(gca,'ColorOrderIndex',1)
plot(z')
legend('xref', 'yref', 'thref', 'x', 'y', 'th')
title('Z')

subplot(1,2,2)
hold on
plot(v_ref', '--')
set(gca,'ColorOrderIndex',1)
plot(v')
legend('vxref', 'vyref', 'wzref', 'vx', 'vy', 'wz')
title('V')