clc
clear
close all

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



% % Generate obstacle points along the hemisphere boundaries for bug algorithm
% numObstaclePoints = 100;  % Number of points to represent each hemisphere
% leftObstacles = generateHemisphereObstacles(center1X, centerY, radius+0.5, pi/2, -pi/2, numObstaclePoints);
% rightObstacles = generateHemisphereObstacles(center2X, centerY, radius+0.5, pi/2, 3*pi/2, numObstaclePoints);
% obstaclePoints = [leftObstacles, rightObstacles]; 

%% Define the robots
r_a = 1;
n = 0;
samples = 5000;
detection_radius = 6;
sampleTime = 0.1;

record_trajctories = struct;
recorded_samples = {};

animation = false;
record_video = false;
record_data = true;

if animation
    h_fig = figure('Name', 'Robot Navigation Animation', 'Position', [100, 100, 600, 1200]);
    set(gcf, 'DoubleBuffer', 'on');
    set(gcf, 'Color', 'white');
end

if record_video
    videoFilename = 'narrow_corridor_simulation.mp4';
    videoObj = VideoWriter(videoFilename, 'MPEG-4');
    videoObj.FrameRate = 30;  % Set the frame rate (30 fps is standard)
    videoObj.Quality = 100;   % Set the quality (100 is highest)
    open(videoObj);
    
    % Create the figure with higher resolution for better video quality
    

    frameCount = 0;
end

while n < samples
    robot2_start = generateRandomCoordinate(0, 15, 0, 3);
    robot2_goal = generateRandomCoordinate(0, 15, 27, 30);
    robot3_start = generateRandomCoordinate(0, 15, 27, 30);
    robot3_goal = generateRandomCoordinate(0, 15, 0, 3);

    robot2 = MbRobot;
    robot2.name = 'Ally';
    robot2.Start = robot2_start;
    robot2.Goal = robot2_goal;
    robot2.SafeRangeColor = [0.8 1 0.8 1];
    robot2.DetectionRangeColor = [0.8 1 0.8 0.1];
    robot2.SafeRadius = r_a;
    robot2.HeadAngle = atan2(robot2.Goal(2) - robot2.Start(2), ...
                                robot2.Goal(1) - robot2.Start(1));
    robot2.CurrentPose = [robot2.Start; robot2.HeadAngle; robot2.AngularVelocity];
    robot2.CurrentCoord = robot2.CurrentPose(1:2);
    robot2.DetectionRadius = detection_radius;
    robot2.StaticObstacle = StaticObstacleInfo;


    robot3 = MbRobot;
    robot3.name = 'Adversarial';
    robot3.Start = robot3_start;
    robot3.Goal = robot3_goal;
        
    robot3.SafeRangeColor = [0.6 0.8 1 1];
    robot3.DetectionRangeColor = [0.6 0.8 1 0.1];
    robot3.SafeRadius = r_a;
    robot3InitPose = robot3.CurrentPose;
    robot3.HeadAngle = atan2(robot3.Goal(2) - robot3.Start(2), ...
                                robot3.Goal(1) - robot3.Start(1));
    robot3.CurrentPose = [robot3.Start; robot3.HeadAngle; robot3.AngularVelocity];
    robot3.CurrentCoord = robot3.CurrentPose(1:2);
    robot3.DetectionRadius = detection_radius;
    robot3.StaticObstacle = StaticObstacleInfo;


    record_trajctories.robot2.start = robot2.Start;
    record_trajctories.robot2.goal = robot2.Goal;
    record_trajctories.robot3.start = robot3.Start;
    record_trajctories.robot3.goal = robot3.Goal;
    

    robot2Poses = [];
    robot2Velocities = [];
  
    robot3Poses = [];
    robot3Velocities = [];

    interaction_state = struct();
    interaction_state.priority_decided = true;
    interaction_state.robot2_avoiding = true;
    interaction_state.robot3_avoiding = false;

    % figure('Name', 'Robot Navigation Animation', 'Position', [100, 100, 500, 1000]);
    % set(gcf, 'DoubleBuffer', 'on');  % Enable double buffering for smoother animation
    % 

    heading_smoothing_factor = 0.9;
    for t = 0:sampleTime:50
        robot2.detection_indicator = 0;
        robot3.detection_indicator = 0;
        % Check if both robots reached their goals
        if robot2.atGoal() && robot3.atGoal()
            fprintf('Sample %d, both robots reached their goals!\n', n+1);
            record_trajctories.colide = false;
            break;
        end

        if calcDist(robot2.CurrentCoord, robot3.CurrentCoord) < robot2.SafeRadius
            fprintf('Sample %d collided!\n', n+1);
            record_trajctories.colide = true;
            break;
        end
        
        % Store previous states
        if ~robot2.atGoal()
            previousHeading2 = robot2.HeadAngle;
        else
            robot2.DefaultLinearVelocity = 0;
        end
        
        if ~robot3.atGoal()
            previousHeading3 = robot3.HeadAngle;
        else
            robot3.DefaultLinearVelocity = 0;
        end
        
        % Calculate if robots can detect each other
        robotsCanDetectEachOther = calcDist(robot2.CurrentCoord, robot3.CurrentCoord) <= robot2.DetectionRadius;
        
        % STEP 1: Determine robot behavior based on proximity
        if robotsCanDetectEachOther
            % if ~interaction_state.priority_decided
            %     % Use geometric properties to determine priority
            % 
            %     % 1. Calculate progress along intended paths
            %     robot2_progress = calcProgressAlongPath(robot2.Start, robot2.Goal, robot2.CurrentCoord);
            %     robot3_progress = calcProgressAlongPath(robot3.Start, robot3.Goal, robot3.CurrentCoord);
            %     % Assign roles based on priority score
            %     if robot2_progress < robot3_progress
            %         interaction_state.robot2_avoiding = true;
            %         interaction_state.robot3_avoiding = false;
            %     else
            %         interaction_state.robot3_avoiding = true;
            %         interaction_state.robot2_avoiding = false;
            %     end
            %     interaction_state.priority_decided = true;
            % end
            
            if ~robot2.atGoal()
                if interaction_state.robot2_avoiding
                    dynamic_obstacle = robot3.CurrentCoord;
                    robot2.APF_Dynamic(sampleTime, 1, 0.6, dynamic_obstacle);
                    robot2.detection_indicator = 1;
                else
                    robot2.APF_Static(sampleTime, 1, 0.4);
                    robot2.detection_indicator = 0;
                end
                robot2.HeadAngle = heading_smoothing_factor * previousHeading2 + (1-heading_smoothing_factor) * robot2.HeadAngle;
            end
            
            if ~robot3.atGoal()
                if interaction_state.robot3_avoiding
                    dynamic_obstacle = robot2.CurrentCoord;
                    robot3.APF_Dynamic(sampleTime, 1, 0.6, dynamic_obstacle);
                    robot3.detection_indicator = 1;
                else
                    robot3.APF_Static(sampleTime, 1, 0.4);
                    robot3.detection_indicator = 0;
                end
                robot3.HeadAngle = heading_smoothing_factor * previousHeading3 + (1-heading_smoothing_factor) * robot3.HeadAngle;
            end
        
        % STEP 3: No interaction - both use static planning
        else
            % Reset interaction state when agents move out of range
            interaction_state.priority_decided = false;
            % Robot 2 uses static planning
            if ~robot2.atGoal()
                robot2.APF_Static(sampleTime, 1, 0.4);
                robot2.HeadAngle = heading_smoothing_factor * previousHeading2 + (1-heading_smoothing_factor) * robot2.HeadAngle;
            end
            
            % Robot 3 uses static planning
            if ~robot3.atGoal()
                robot3.APF_Static(sampleTime, 1, 0.4);
                robot3.HeadAngle = heading_smoothing_factor * previousHeading3 + (1-heading_smoothing_factor) * robot3.HeadAngle;
            end
        end
        
        % STEP 4: Update positions and stay within boundaries
        if ~robot2.atGoal()
            robot2.CurrentCoord(1) = max(0, min(envWidth, robot2.CurrentCoord(1)));
            robot2.CurrentCoord(2) = max(0, min(envHeight, robot2.CurrentCoord(2)));
            robot2Poses = [robot2Poses, [robot2.CurrentPose; robot2.detection_indicator]];
            robot2Velocities = [robot2Velocities, robot2.velocity];
        end
        
        if ~robot3.atGoal()
            robot3.CurrentCoord(1) = max(0, min(envWidth, robot3.CurrentCoord(1)));
            robot3.CurrentCoord(2) = max(0, min(envHeight, robot3.CurrentCoord(2)));
            robot3Poses = [robot3Poses, [robot3.CurrentPose; robot3.detection_indicator]];
            robot3Velocities = [robot3Velocities, robot3.velocity];
        end

       
        %%%% Need to infer the location based on the trajectory
        %%%% Could set different decrease rate
        %%%% Still use prediction accuracy to do examine the result
        %%%% Assume map info is known to us so we can use the central of
        %%%% the hemisphere

        %%%% APF, add another layer of hard constraint, to force the robot
        %%%% to turn to another safe direction

        if animation
            
            clf(h_fig, 'reset'); % clear current figure
            
            hold on;
            % scatter(obstaclePoints(1,:), obstaclePoints(2,:));
            % plot(robot1.Start(1),robot1.Start(2),'go','MarkerSize',10,'MarkerFaceColor',[0.8500 0.3250 0.0980])
            % plot(robot1.Goal(1),robot1.Goal(2),'rp','MarkerSize',10,'MarkerFaceColor',[0.8500 0.3250 0.0980])
            plot(robot2.Start(1),robot2.Start(2),'go','MarkerSize',10,'MarkerFaceColor','g')
            plot(robot2.Goal(1),robot2.Goal(2),'rp','MarkerSize',10,'MarkerFaceColor','g')
            plot(robot3.Start(1),robot3.Start(2),'go','MarkerSize',10,'MarkerFaceColor','b')
            plot(robot3.Goal(1),robot3.Goal(2),'rp','MarkerSize',10,'MarkerFaceColor','b')

            if ~isempty(robot2Poses)
                plot(robot2Poses(1,:),robot2Poses(2,:),'-',...
                    'Color',[0,255,0]/255,'LineWidth',2.5);
            end

            if ~isempty(robot3Poses)
                plot(robot3Poses(1,:),robot3Poses(2,:),'-',...
                    'Color',[0,0,255]/255,'LineWidth',2.5);
            end

            plotCorridorSimulation(envWidth, envHeight, passageWidth)
            robot2.plotRobot(1); % adjust the frame size as needed
            robot3.plotRobot(1);

            

            drawnow;

            if record_video
                pause(0.001);
                frame = getframe(h_fig);
                writeVideo(videoObj, frame);
                
                frameCount = frameCount + 1;
            end
    

        end


    end

    record_trajctories.robot2.poses = robot2Poses;
    record_trajctories.robot2.velocities = robot2Velocities;
    record_trajctories.robot3.poses = robot3Poses;
    record_trajctories.robot3.velocities = robot3Velocities;
    
    n = n + 1;

    recorded_samples{n} = record_trajctories;
end

if record_video
    frame = getframe(gcf);
    writeVideo(videoObj, frame);
end

if animation && record_video
    close(videoObj);
    fprintf('Video saved as "%s"\n', videoFilename);
end

if record_data
    save('Data/SimulationData/Narrow Corridor Simulation.mat', "recorded_samples")
end


%% Functions
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
    plot([0, 0], [0, envHeight], 'k-', 'LineWidth', 2);
    plot([envWidth, envWidth], [0, envHeight], 'k-', 'LineWidth', 2);
    % Top and bottom horizontal boundaries
    plot([0, envWidth], [0, 0], 'k-', 'LineWidth', 2);
    plot([0, envWidth], [envHeight, envHeight], 'k-', 'LineWidth', 2);
    
    % Plot the hemispheres (obstacles)
    % Left hemisphere
    theta = linspace(pi/2, -pi/2, 100);  % Only right half of the circle
    leftHemisphereX = center1X + radius * cos(theta);
    leftHemisphereY = centerY + radius * sin(theta);
    fill(leftHemisphereX, leftHemisphereY, 'k');
    
    % Right hemisphere
    theta = linspace(pi/2, 3*pi/2, 100);  % Only left half of the circle
    rightHemisphereX = center2X + radius * cos(theta);
    rightHemisphereY = centerY + radius * sin(theta);
    fill(rightHemisphereX, rightHemisphereY, 'k');
    
    
end

function coord = generateRandomCoordinate(xMin, xMax, yMin, yMax)
    % Set default parameters if not provided
    if nargin < 1
        xMin = 0;
    end
    if nargin < 2
        xMax = 15;
    end
    if nargin < 3
        yMin = 0;
    end
    if nargin < 4
        yMax = 8;
    end
    
    % Generate random x within the range [xMin, xMax]
    x = xMin + (xMax - xMin) * rand();
    
    % Generate random y within the range [yMin, yMax]
    y = yMin + (yMax - yMin) * rand();
    
    coord = [x;y];
end


function obstacles = generateHemisphereObstacles(centerX, centerY, radius, startAngle, endAngle, numPoints)
    % Generate points along the hemisphere boundary to be used as obstacles
    angles = linspace(startAngle, endAngle, numPoints);
    obstacles = zeros(2, numPoints);
    for i = 1:numPoints
        obstacles(1, i) = centerX + radius * cos(angles(i));
        obstacles(2, i) = centerY + radius * sin(angles(i));
    end
end


function progress = calcProgressAlongPath(start, goal, current)
    path_vector = goal - start;
    path_length = norm(path_vector);
    
    % Vector from start to current position
    current_vector = current - start;
    
    % Project current position onto path
    projection = dot(current_vector, path_vector/path_length);
    
    % Normalize to get progress (0-1)
    progress = min(max(projection/path_length, 0), 1);
end