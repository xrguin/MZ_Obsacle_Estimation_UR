clc
clear
close all

%%
load('Data/SimulationData/Narrow Corridor Simulation.mat','recorded_samples')
doTrain = false;
num_samples = size(recorded_samples,2) - 200;
window_len = 10;

if doTrain
    
    XAll = {};
    YAll = {};
    k = 1;
    for i = 1:num_samples
        sample = recorded_samples{i};
        detected_cols = sample.robot2.poses(end,:) == 1;
        % Find transitions
        transitions = diff(detected_cols);
        % Start indices of sequences of 1s
        start_indices = find(transitions == 1);
        % End indices of sequences of 1s
        end_indices = find(transitions == -1) - 1;
    
        
        robot2AvoidTraj = sample.robot2.poses(1:3,start_indices - window_len:end_indices)';
        robot3AvoidTraj = sample.robot3.poses(1:2,detected_cols)';
    
        num_windows = size(robot3AvoidTraj,1);
        for j = 1:num_windows
    
            XAll{k,1} = robot2AvoidTraj(j:j+window_len-1,:);
            YAll{k,1} = robot3AvoidTraj(j,:);
            
            k = k+1;
        end
    end
    
    %%
    randOrder = randperm(size(XAll,1));
    XAll_shuffled = XAll(randOrder);
    YAll_shuffled = YAll(randOrder);
    
    totalSamples = size(XAll,1);
    iTrain = 1:floor(0.8*totalSamples);
    iVal = iTrain(end) : totalSamples;
    
    
    XTrain = XAll_shuffled(iTrain,1);
    XVal = XAll_shuffled(iVal,1);
    
    
    YTrain = cell2mat(YAll_shuffled(iTrain,1));
    YVal = cell2mat(YAll_shuffled(iVal,1));
    
    
    %%
    numChannels = 3;
    numOutputs = 2;

    layers = [...
        sequenceInputLayer(numChannels)
        % lstmLayer(100,'OutputMode','sequence')
        % % lstmLayer(100,'OutputMode','sequence')
        % dropoutLayer(0.2)
        % lstmLayer(75, "OutputMode","sequence")
        % dropoutLayer(0.25)
        lstmLayer(32, 'OutputMode', 'sequence')
        dropoutLayer(0.2)
        lstmLayer(128, 'OutputMode', 'last')
        fullyConnectedLayer(numOutputs)
        ];

    % layerGraph(layers)
    options = trainingOptions('adam', ...
        'ExecutionEnvironment','gpu',...
        MaxEpochs=20, ...
        MiniBatchSize=128,...
        InitialLearnRate=0.001,...
        LearnRateSchedule='piecewise',...
        LearnRateDropPeriod=100,...
        LearnRateDropFactor=0.2,...
        SequencePaddingDirection='left',...
        Shuffle='every-epoch',...
        GradientThreshold=0.5,...
        Verbose=true,...
        Plots='training-progress',...
        ValidationData={XVal,YVal});
    %% Train Recurrent Neural Network
    multiSliceNet = trainnet(XTrain, YTrain, layers, 'mse',options);
    save('Data/Network/Narrow Corridor Net.mat','multiSliceNet')
end

%%
% Define figure size and spacing
figWidth = 500;   % Width of each figure in pixels
figHeight = 1000;  % Height of each figure in pixels
spacing = 50;     % Space between figures in pixels
screenSize = get(0, 'ScreenSize'); % Get screen size: [left, bottom, width, height]

% Calculate positions
leftPos1 = (screenSize(3) - 2*figWidth - spacing) / 2;
bottomPos = (screenSize(4) - figHeight) / 2;
leftPos2 = leftPos1 + figWidth + spacing;

%%
load('Data/Network/Narrow Corridor Net.mat','multiSliceNet')
testIdx = 30;
sample = recorded_samples{num_samples + testIdx};
detected_cols = sample.robot2.poses(end,:) == 1;
% Find transitions
transitions = diff(detected_cols);
% Start indices of sequences of 1s
start_indices = find(transitions == 1);
% End indices of sequences of 1s
end_indices = find(transitions == -1) - 1;


robot2AvoidTraj = sample.robot2.poses(1:3,start_indices - window_len:end_indices)';
robot3AvoidTraj = sample.robot3.poses(1:2,detected_cols)';

num_windows = size(robot3AvoidTraj,1);
j = 1;

Y_norm= nan(num_windows,2); % Out of plotting range
Y_reg = Y_norm;

for t = 1:num_windows-1

    Xmoving_window = robot2AvoidTraj(t:t+window_len-1,1:3);
    Ymoving_window = robot3AvoidTraj(t,1:2);
    XTrajTest{j} = Xmoving_window;
    YTrajTest{j} = Ymoving_window;
    [Y_reg(t,:),state] = predict(multiSliceNet,XTrajTest{j});
    multiSliceNet.resetState();
    j = j + 1;
end

%%

figure('Name','Results','Position',[leftPos1, bottomPos, figWidth, figHeight]);
c = linspace(1,10, length(Y_reg));
scatter(robot3AvoidTraj(:,1), robot3AvoidTraj(:,2),[],c, ...
    'DisplayName','Actual Enemy Locations');
hold on
% 
% scatter(enemydata(offset+section_start,1),enemydata(offset+section_start,2),80,'magenta','filled','o', ...
%     'DisplayName','Non-straight Starting Point')
% plot(allydata(1:offset,1), allydata(1:offset,2),'b','LineWidth',2,'DisplayName','Ally')
scatter(robot3AvoidTraj(1,1), robot3AvoidTraj(1,2),60,'filled','d', ...
    'DisplayName','Enemy Start Point','MarkerFaceColor',[0.4940 0.1840 0.5560])
plot(robot2AvoidTraj(:,1),robot2AvoidTraj(:,2),'Color',[0 0.4470 0.7410],'LineWidth',2,'DisplayName','Ally Trajectory')
plot(robot3AvoidTraj(:,1),robot3AvoidTraj(:,2),'--','DisplayName','Enemy Trajectory')
% plot(allydata(1:offset,1), allydata(1:offset,2),'g','LineWidth',2, ...
%     'DisplayName','First Input (Ally)')
scatter(robot2AvoidTraj(1,1),robot2AvoidTraj(1,2),60,'filled','d','DisplayName','Ally Start Point','MarkerFaceColor',[0.8500 0.3250 0.0980])
scatter(Y_reg(:,1), Y_reg(:,2),[],c,'filled','DisplayName','Predicted Enemy Locations')
% first_not_straight = XTrajTest{section_start(1)};
% plot(first_not_straight(2:end,1), first_not_straight(2:end,2),'g','LineWidth',2, ...
%     'DisplayName','First Non-straint Input Data')
% scatter(Y_reg(section_start,1),Y_reg(section_start,2),80,'red','filled','o', ...
%     'DisplayName','Non-straight Starting Point')
xlabel("X")
ylabel('Y')
pbaspect([1 2 1])
colorbar;
colormap("parula");
cb = colorbar;
cb.Label.String = 'Predict Steps';
cb.Limits = [1,10];
cb.Ticks = [1 10];  % Place ticks at the ends of the color bar
cb.TickLabels = {'1', num_windows};  % Set the labels at these ticks to '1' and '92'
legend('show')
ax = gca;
ax.FontSize = 16;
hold off


%% Define Corridor Parameters

envWidth = 15;
envHeight = 30;
passageWidth = 2;

radius = (envWidth - passageWidth) / 2;
center1X = 0;
center2X = envWidth;
centerY = envHeight / 2;

corridor_center = [(center1X + center2X)/2; centerY];


theta = linspace(pi/2, -pi/2, 100);  % Only left half of the circle
leftHemisphereX = center1X + radius * cos(theta);
leftHemisphereY = centerY + radius * sin(theta);

theta = linspace(pi/2, 3*pi/2, 100);  % Only right half of the circle
rightHemisphereX = center2X + radius * cos(theta);
rightHemisphereY = centerY + radius * sin(theta);

StaticObstacleInfo = NaN(3,2);
StaticObstacleInfo(1:2,1) = [center1X, centerY]';
StaticObstacleInfo(3,1) = radius;
StaticObstacleInfo(1:2,2) = [center2X, centerY]';
StaticObstacleInfo(3,2) = radius;

%% Define the robots
r_a = 1;
n = 0;
samples = 1;
detection_radius = 6;
sampleTime = 0.1;

animation = true;
record_video = false;

h_fig = figure('Name', 'Robot Navigation Animation', 'Position', [leftPos2, bottomPos, figWidth, figHeight]);
set(gcf, 'DoubleBuffer', 'on');
set(gcf, 'Color', 'white');

% sz = window_len;
% c = linspace(1,10,size(current_input,1));
% scatter(x,y,sz,c,'filled')

if record_video
    videoFilename = 'narrow_corridor_prediction.mp4';
    videoObj = VideoWriter(videoFilename, 'MPEG-4');
    videoObj.FrameRate = 30;  % Set the frame rate (30 fps is standard)
    videoObj.Quality = 100;   % Set the quality (100 is highest)
    open(videoObj);
    
    % Create the figure with higher resolution for better video quality
    

    frameCount = 0;
end


if animation
for steps = 1:size(Y_reg,1)
    clf(h_fig, 'reset'); % clear current figure
    hold on;
    plotCorridorSimulation(envWidth, envHeight, passageWidth)
    current_input = robot2AvoidTraj(steps:steps+window_len-1,1:2);
    current_output = Y_reg(steps,:);

    c = linspace(1,10,size(current_input,1));
    
    scatter(current_input(:,1), current_input(:,2),400,c,'DisplayName','Robot2 Locations')
    % plotDetectionRange(current_output(1), current_output(2),1,[1 0 0 0.5])
    % plotDetectionRange(robot3AvoidTraj(steps,1), robot3AvoidTraj(steps,2),1,[0 1 0 0.5])
    scatter(current_output(1), current_output(2),400,'red','filled','DisplayName','Predicted Robot3 Location')
    scatter(robot3AvoidTraj(steps,1), robot3AvoidTraj(steps,2),400,"green",'filled','DisplayName','Actual robot3 Location')
    legend('show')
    set(gca, 'FontSize', 15)

    pause(0.001);

    if record_video
        frame = getframe(h_fig);
        writeVideo(videoObj, frame);
        
        frameCount = frameCount + 1;
    end

end
end


if record_video
    frame = getframe(gcf);
    writeVideo(videoObj, frame);
end

if animation && record_video
    close(videoObj);
    fprintf('Video saved as "%s"\n', videoFilename);
end


%%
function plotCorridorSimulation(envWidth, envHeight, passageWidth)
   
    
    % Hemisphere parameters
    radius = (envWidth - passageWidth) / 2;
    center1X = 0;
    center2X = envWidth;
    centerY = envHeight / 2;
    
    % Create plot area with appropriate limits
    hold on;
    axis([0, envWidth, 0, envHeight]);
    title('Narrow Corridor Environment');
    xlabel('X');
    ylabel('Y');
    
    % Plot boundaries
    % Left and right vertical boundaries
    % plot([0, 0], [0, envHeight], 'k-', 'LineWidth', 2);
    % plot([envWidth, envWidth], [0, envHeight], 'k-', 'LineWidth', 2);
    % % Top and bottom horizontal boundaries
    % plot([0, envWidth], [0, 0], 'k-', 'LineWidth', 2);
    % plot([0, envWidth], [envHeight, envHeight], 'k-', 'LineWidth', 2);
    % Plot boundaries with no legend
    h1 = plot([0, 0], [0, envHeight], 'k-', 'LineWidth', 2);
    h2 = plot([envWidth, envWidth], [0, envHeight], 'k-', 'LineWidth', 2);
    h3 = plot([0, envWidth], [0, 0], 'k-', 'LineWidth', 2);
    h4 = plot([0, envWidth], [envHeight, envHeight], 'k-', 'LineWidth', 2);

    % Disable legend display for walls
    set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h3,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    set(get(get(h4,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');

    
    % Plot the hemispheres (obstacles)
    % Left hemisphere
    theta = linspace(pi/2, -pi/2, 100);  % Only right half of the circle
    leftHemisphereX = center1X + radius * cos(theta);
    leftHemisphereY = centerY + radius * sin(theta);
    % fill(leftHemisphereX, leftHemisphereY, 'k');
    h5 = fill(leftHemisphereX, leftHemisphereY, 'k');
    set(get(get(h5,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    
    % Right hemisphere
    theta = linspace(pi/2, 3*pi/2, 100);  % Only left half of the circle
    rightHemisphereX = center2X + radius * cos(theta);
    rightHemisphereY = centerY + radius * sin(theta);
    % fill(rightHemisphereX, rightHemisphereY, 'k');
    h6 = fill(rightHemisphereX, rightHemisphereY, 'k');
    set(get(get(h6,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    
    
end


function plotDetectionRange(centerX, centerY, r, color)

rectangle('Position',[centerX-r, ...
                centerY-r, 2*r, 2*r],...
                'Curvature',[1,1], 'EdgeColor','none', ...
                'FaceColor',color, ...
                'FaceAlpha',color);
end