function displayState(filename, state)

%fid = fopen('C:\Users\gorune\Desktop\pde data.dat');
fid = fopen(filename);

dims = fscanf(fid, '%i,%i,%f,%f,%f,%f,%i.', 7);

cellsX = dims(1);
cellsY = dims(2);
startX = dims(3);
endX   = dims(4);
startY = dims(5);
endY   = dims(6);
numStates = dims(7);

solutionT = zeros(cellsX, cellsY, numStates);
solution  = zeros(cellsY, cellsX, numStates);
for i = 1:numStates
    solutionT(:,:,i) = fscanf(fid, '%g', [cellsX cellsY]);
end
fclose(fid);

% Transpose so that A matches
% the orientation of the file
for i = 1:numStates
    solution(:,:,i) = solutionT(:,:,i)';
end

rangeX = startX:(endX-startX)/cellsX:endX-(endX-startX)/cellsX;
rangeY = startY:(endY-startY)/cellsY:endY-(endY-startY)/cellsY;
surf(rangeX, rangeY, solution(:,:,state))
xlabel('x');
ylabel('y');
zlabel('state');
axis 'equal';