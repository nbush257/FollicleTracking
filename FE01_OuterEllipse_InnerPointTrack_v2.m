clear;clc;
textOffset = 40;  % offset for writing BadInner BadOuter text;  Manually chosen.
%%% Choose all TIF numbers (Sections) you want to work on
%%% Note: If you keep previous sections in this array that you've already finished 
%%% it will quickly plot through all of them before getting to the section you want.  
%%% (it will stop at the 
%%% while loop when it discovers all of the follicles are done. You will
%%% probably want to update the tif number you start with each time you
%%% restart the program.
% tifNumbers = [6318:6326];
% tifNumbers = [6369:6373];
% tifNumbers = [6452:6456];
% tifNumbers = [6375:10:6515];
tifNumbers = [6385];
for qq = 1:length(tifNumbers)   %% loop through the Sections
    clear nFollicles Ellipses* Outer* Inner* fols2plot BadOuters BadInners;
    fols2plot=[];  % used only for plotting for user display in figs 1 and 2
    fols2plotI=[]; % used only for plotting for user display in figs 1 and 2
    BadOuters = [];   % keeps track of all bad follicles
    BadInners = [];  % keeps track of all follicles where ONLY the inner ellipse is bad
    tifNumber = tifNumbers(qq);
    fname = (['Pad2_' int2str(tifNumber) '.tif']);
    abc = imread(fname);
    b = abc(:,:,2);  % select the green channel
    
    %%% Fig 1 will be the figure where the user can watch what's happening.
    fig1;
    clf;imagesc(abc);  hold on;  
    h =  title(['Section ' int2str(tifNumber)]);set(h,'FontSize',14);
    %%% Load up all the follicles that have already been tracked for this
    %%% section (if there are any)  Put them into fols2plot and also start
    %%% making the BadOuters and BadInners list for this section. 
    s = dir(['*' int2str(tifNumber) '_fol*.mat' '']);
    maxfol = 0;
    if ~isempty(s)
        for ii = 1:length(s)
            data = s(ii).name;
            load(data);
            fols2plot{ii} = EllipsesO;
            fols2plotI{ii} = EllipsesI;
            BadOuters(ii) = BadOuter;
            BadInners(ii) = BadInner;
            mf = eval(data(end-5:end-4));
            if mf >maxfol
                maxfol = mf;
            end;
        end;
    else
        maxfol = 0;
    end;
    %%% This if statement is a paranoid sanity check
    %%% They should always be equal
    if maxfol ~= length(s)
        disp('error in fol number');
    end;
    %%% Plot the outer ellipses in figure 1 so user can see how 
    %%% far along we are.
    if ~isempty(fols2plot)
        for ii = 1:length(fols2plot)
            data = fols2plot{ii};
            plotv(data,'y.');
            data = fols2plotI{ii};
            plotv(data,'g.');
            if BadOuters(ii) ==1
                h = text(mean(data(1,:))-textOffset,mean(data(2,:)),'BO'); set(h,'color','y');
            end;
            if BadInners(ii) ==1
                h = text(mean(data(1,:)+textOffset),mean(data(2,:)),'BI'); set(h,'color','g');
            end;
        end;
    end;
     
    %%% Now we're going to start work on the next follicle number. 
    %%% Need to be sure to clear out any old Ellipse variables (as well as
    %%% the raw data, which are stored in OuterRawPts and InnerRawPts)
    %%% because when we loaded up the follicles that had already been
    %%% tracked, those came along for the ride.  We need to be sure to
    %%% clear them out. 
    folNumber = maxfol + 1;
    clear Ellipses* Outer* Inner* BadOuter BadInner
    
    %%% If the user's already been tracking follicles in this section then we already 
    %%% know how many follicles there were, no need to ask
    %%% again.  That number (nFollicles) got loaded up with the other previous follicles. 
    if exist('nFollicles') == 0
        nFollicles = input('Input number of Follicles: ');
    end;
    
    %%%  Here's the start of the big loop on all the follicles in the
    %%%  section.
    while folNumber <= nFollicles    
        %%% again, be sure to clear out variables from the previous
        %%% follicle. 
        clear Ellipse* Outer* Inner* BadOuter BadInner
        disp(['Working on follicle number ' int2str(folNumber)]);
        %%% Plot the identical image in Figure 2 as was plotted in Figure
        %%% 1 in order to allow the user to select the relevant follicle
        %%% with a big image. Figure 2 will soon turn into something
        %%% different, but for now we want the user to select the follicle
        %%% in figure 2.   That way the user doesn't have to go back and
        %%% forth between Figure 1 and Figure 2. Figure 1 is just to
        %%% display progress only, so user can see how things are going overall.
        fig2;
        clf;
        imagesc(b);colormap('gray'); hold on;
        h =  title(['Section ' int2str(tifNumber)]);set(h,'FontSize',14);
        for ii = 1:length(fols2plot)
            data = fols2plot{ii};
            plotv(data,'y');
        end;
        
        %%% The user is now going to click on 2 points to establish the 
        %%% region the follicle is in.  The points dont have to be precise,
        %%% though if you click right on the follicle the threshold will be
        %%% closer to correct at the start. 
        [x,y]= myginput(2,'arrow');
        x  = round(x); y = round(y);
        %%% The user probably clicked too close to the follicle (or maybe
        %%% even on the follicle) so give the code some wiggle boundary
        %%% room around the follicle. 
        % sf is the  scale factor for boundary box around input points
        sf = 1.1;   
        % init means initial because these are the x and y values for the
        % boundary box in the initial part of the code.  It's not a great
        % name.  But I wanted distinct variable names because we're going
        % to need them again at the very end of this code when we have to
        % put the tracked points back in the original figure. 
        xINIT = round(min(y)/sf):round(max(y)*sf);   
        yINIT = round(min(x)/sf):round(max(x)*sf);
        %%%  b1 is going to be just that rectangular portion of the image b
        %%%  that the user selected. 
        b1 =  b(round(min(y)/sf):round(max(y)*sf),round(min(x)/sf):round(max(x)*sf));
        
        %%% Here's where figure 2 does its major change.  The big image
        %%% where the user selected points disappears and is replaced with
        %%% a set of subplots that allow the user to look at the
        %%% consequences of varying the threshold on the fitting to the
        %%% inner and outer ellipse.
        %%% In the first subplot we just plot the raw image of the
        %%% follicle. This never changes, so the user can see the raw
        %%% follicle. 
        fig2;
        clf;
        subplot(3,2,1);imagesc(b1);colormap('gray');
        h =  title(['Section ' int2str(tifNumber)]);set(h,'FontSize',10);

        %%% In the second subplot we're going to plot how the image looks
        %%% when we turn it black and white using the threshold established
        %%% by where the user clicked. Recall from line 98 that the user
        %%% clicked on points x and y in image b (not b1, which is the small
        %%% rectangle), so find the intensity values there 
        %%% 
        for ii = 1:length(x)
            tval(ii) = b(y(ii),x(ii));   % tval stands for threshold value.
        end;
        b2 = b1;   % b2 is going to be the black-white version of b1.
        b2(b1>mean(tval)) = 1; 
        b2(b1<mean(tval)) = 0; 
        % turn white to black and black to white
        % not quite sure why I needed the additional variable bw.  I could
        % contineud to have called it b2, but it seemed like a good idea
        % to name it bw at the time.  bw is now taking over for b2 and we're not going to
        % use b2 again. 
        bw = imbinarize(abs (double(b2) - 1)); 
        %%% plot bw and add the labels that tell user where to go up and
        %%% down.
        subplot(3,2,2); imagesc(bw);hold on;
        dum = axis;
        h=text(10,30,'UP'); set(h,'color','y','FontSize',14);
        h=text(dum(2)/1.2,30,'DOWN'); set(h,'color','g','FontSize',14);
        
        %%% Okay, so this now gets a little tricky. 
        %%% UserHappyWithResults means that the follicle is finished and
        %%% can be saved and plotted on the image in Figure 1. 
        %%% UserHappy means that the user is happy with this particular
        %%% iteration of the follicle threshold.   The user might want to
        %%% try a couple of iterations to get the threshold right. 
        
        %%%  User starts off unhappy with follicle fitting (it has not been
        %%%  fit yet)
        UserHappyWithResults = 0;   
        while UserHappyWithResults == 0
            %%%   Is the user happy with this *iteration* of the threshold?
            UserHappy = 0;   
            while UserHappy == 0;
                %%% The UP and DOWN text above was placed at arbitrary
                %%% positions.  Put some asterisks so user can really see
                %%% where the clicking needs to happen. 
                dum = axis;
                plot(dum(2)/5,dum(4)/5,'y*');
                plot(4*dum(2)/5,dum(4)/5,'g*');
                disp('click on follicle wall or adjust threshold?');
                [xr,yr]= myginput(1,'arrow');
                % if click is in the top left corner, adjust threshold up
                if xr < dum(2)/5 & yr <dum(4)/5   
                    tval = tval*1.1;
                    b2 = b1;
                    b2(b1>mean(tval)) = 1;  
                    b2(b1<mean(tval)) = 0;  
                    bw = imbinarize(abs (double(b2) - 1));
                    subplot(3,2,2); cla; imagesc(bw);hold on;
                    h=text(10,30,'UP'); set(h,'color','y','FontSize',14);
                    h=text(dum(2)/1.2,30,'DOWN'); set(h,'color','g','FontSize',14);
                % if click is in the top right corner, adjust threshold up
                elseif xr > 4*dum(2)/5 & yr <dum(4)/5
                    tval = tval*0.9;
                    b2 = b1;
                    b2(b1>mean(tval)) = 1;  
                    b2(b1<mean(tval)) = 0;  
                    bw = imbinarize(abs (double(b2) - 1));
                    subplot(3,2,2); cla; imagesc(bw);hold on;
                    h=text(10,30,'UP'); set(h,'color','y','FontSize',14);
                    h=text(dum(2)/1.2,30,'DOWN'); set(h,'color','g','FontSize',14);
                %  If user clicked anywhere else then the user is happy
                %  enough with this threshold choice to try this iteration.
                % Note that user must click on follicle wall in order for
                % code to find follicle wall. 
                else
                    UserHappy =1;
                end;
            end;
            
            %%%% Here is a really key point -- the user has to have clicked
            %%%% on the follicle wall.  The code is going to choose what is
            %%%% the "follicle wall" based on where the user clicked.  All
            %%%% of the different white regions get thrown into PixelList
            %%%% and then we find the white region closest to the pixel
            %%%% that the user clicked on.
            stats = regionprops(bw,'PixelList');
            xr = round(xr); yr = round(yr);
            
            %%% Minimize the distance between all the regions found by 
            %%% regionprops in the image and the location where the user
            %%% clicked. 
            counter = 0;
            minDistVal = inf;
            winner = [];
            for ii = 1:length(stats)
                data = stats(ii).PixelList;
                d = sqrt( (data(:,1)-xr).^2 + (data(:,2)-yr).^2 );
                d = min(d);
                if d < minDistVal
                    winner = ii;
                    minDistVal = d;
                end;
            end;
            
            %%% in the variable bw2 we're going to put only the "winning"
            %%% white region that the user said was the follicle wall. bw2
            %%% is totally empty except for the pixels contained in that
            %%% follicle wall region.  Plot bw2 in the third subplot. 
            bw2 = zeros(size(bw));
            data = stats(winner).PixelList;
            % there is a faster way of doing this than
            % a loop.  This is really dumb coding. 
            for ii = 1:length(data)
                bw2(data(ii,2),data(ii,1)) = 1;
            end;
            subplot(3,2,3);imagesc(bw2);ho;
            
            %%% Find the outer edges of the follicle by moving along horizontally
            %%% and finding the upper and lower bounds of the follicle
            %%% wall.  We don't use edge detection becuase we don't know
            %%% what is the outer wall and what's the inner wall
            %%% Remember that the variable "data" is storing the x-y values
            %%% of all of the pixels in the follicle wall, including all the
            %%% inner points, not just the edges.  We'er going to march
            %%% along data horizontally and find its max and min values, which are the 
            %%% upper and lower edges of the outer ellipse. 
            EllipsesO=[];
            counter = 0;
            for ii = 1:size(bw2,2)
                a = find(data(:,1) == ii);
                if ~isempty(a)
                    counter = counter + 1;
                    pt1 = [ii,max(data(a,2))];
                    EllipsesO(counter,1:2) = pt1;
                    counter = counter + 1;
                    pt2 = [ii,min(data(a,2))];
                    EllipsesO(counter,1:2) = pt2;
                end;
            end;
            %%% Define the center of the outer ellipse. 
            EOC = [mean(EllipsesO(:,1)),mean(EllipsesO(:,2))];
            
            %%%% Finding the inner elipse is a bit trickier.  We can still 
            %%% move along horizontally but we need to make an  imaginary cut
            %%% horizontal cut at the midline of the follicle (EOC(2)) and find the
            %%% min values above it and the max values below it.
            %%% Unfortuantely that leaves horizontal lines at the start and
            %%% end of the follicle wall which reflect the follicle wall
            %%% thickness.   We'll get rid of those later.  
            %%% Remember that the variable "data" is storing the x-y values
            %%% of all of the pixels in the follicle wall, including all the
            %%% inner points, not just the edges.  We're going to march
            %%% along data horizontally and find its max and min values above 
            %%% and below the centerline, which are the 
            %%% upper and lower edges of the inner ellipse. 
            EllipsesI=[];
            counter = 0;
            for ii = 1:size(bw2,2)
                a = find(data(:,1) == ii);
                if ~isempty(a)
                    c = find(data(:,1) == ii   & data(:,2) <= EOC(2));
                    if ~isempty(c)
                        pt1 = [ii,max(data(c,2))];
                    else
                        pt1 = [NaN,NaN];
                    end;
                    %          plotv(pt1,'m.');
                    c = find(data(:,1) == ii   & data(:,2) >= EOC(2));
                    if ~isempty(c)
                        pt2 = [ii,min(data(c,2))];
                    else
                        pt2 = [NaN,NaN];
                    end;
                    %          plotv(pt2,'c.');
                    if abs(pt1(2) - pt2(2))>4  & ~isnan(pt1(2)) & ~isnan(pt2(2));
                        counter = counter + 1;
                        EllipsesI(counter,1:2) = pt1;
                        counter = counter + 1;
                        EllipsesI(counter,1:2) = pt2;
                    end;
                end;
            end;
            
            %%% Get rid of horizontal line points at start and end of EllipsesI
            %%% that reflect the width of the follicle.
            %%% We first check that EllipsesI is not empty because if you
            %%% had a very bad threshold chosen it's possible the program
            %%% didn't find any inner ellipse at all and then it will
            %%% crash. 
            %%% Get rid of the horizontal lines by finding points that are
            %%% really far away from the center of the ellipse. 
            if ~isempty(EllipsesI)
                d = sqrt((EOC(1)-EllipsesI(:,1)).^2 + (EOC(2)-EllipsesI(:,2)).^2);
                a = find(d > (mean(d)+2*std(d)));
                EllipsesI(a,:) = [];
                
                %%%% I forget excactly what this part of the code does.  I think
                %%%% that even after we got rid of the horizontal lines the
                %%%% fit to the inner ellipse still wasn't great.  So we
                %%%% fit the ellipse to make a mask  then shrink it based
                %%%% only on where the mask exists *AND* the
                %%%% black-and-white image of the follicle (stored in bw2)
                %%%% is black. 
                [~,el,~,~]= fit_ellipse_mask(EllipsesI(:,1),EllipsesI(:,2));
                if ~isempty(el)
                    bwtest = roipoly(bw2,el(1,:),el(2,:));
                    [yr,xr] = find(bw2 < 0.5 & bwtest > 0.5);
                    EllipsesI2 =[];
                    counter = 0;
                    xvals = min(xr):max(xr);
                    for ii = 1:length(xvals)
                        a = find(xr == xvals(ii));
                        if ~isempty(a)
                            yidx = yr(a);
                            counter = counter + 1;
                            pt1 = [xvals(ii),min(yidx)];
                            %  plotv(pt1,'c*');
                            EllipsesI2(counter,1:2) = pt1;
                            pt2 = [xvals(ii),max(yidx)];
                            %  plotv(pt2,'g*');
                            counter = counter + 1;
                            EllipsesI2(counter,1:2) = pt2;
                        end;
                    end;
                    %%%  EllipsesI2 now holds the best tracking of the inner ellipse we can
                    %%%  get.  Remember though, we haven't done any real
                    %%%  ellipse fitting yet.   These are just the tracked
                    %%%  points.   EllipsesO and EllipsesI hold the tracked
                    %%%  point, not ellipses.
                    EllipsesI = EllipsesI2;
                end % if not isempty(el) from the ellipse fit.
                
                %%% Plot the raw tracked inner and outer points over the bW
                %%% image in subplot 3, and over the original grayscale raw
                %%% image in subplot 5. 
                subplot(3,2,3);ho;
                plot(EllipsesO(:,1),EllipsesO(:,2),'g.');
                h=plot(EllipsesI(:,1),EllipsesI(:,2),'m.');set(h,'LineWidth',12);
                subplot(3,2,5); imagesc(b1);colormap('gray');ho;
                plot(EllipsesO(:,1),EllipsesO(:,2),'g.'); plot(EllipsesI(:,1),EllipsesI(:,2),'m.');
                
                %%% Rename the raw points so we can do the ellipse fitting.
                OuterRawPts = EllipsesO;
                InnerRawPts = EllipsesI;
                
                %%% Now fit the ellipses and plot them in subplots 4 and 6
                [~,el,~,~]= fit_ellipse_mask(EllipsesO(:,1),EllipsesO(:,2));
                if ~isempty(el)
                    subplot(3,2,4);imagesc(bw2);ho; plotv(el,'g.');
                    subplot(3,2,6);imagesc(b1); ho; plotv(el,'g.');
                    EllipsesO = el;
                end;
                
                %%% For the inner wall, don't plot the ellipse, plot the
                %%% tracked points instead.
                [~,el,~,~]= fit_ellipse_mask(EllipsesI(:,1),EllipsesI(:,2));
                if ~isempty(el)
                    subplot(3,2,4); ho; plotv(EllipsesI,'m.');
                    subplot(3,2,6); ho; plotv(EllipsesI,'m.');
                    EllipsesI = el;
                end;
                
            end;  % if ~isempty EllipsesI
            
            %%%% Decide if user is completely happy with results or wants to adjust
            %%%% again through another iteration. Set up the bottom left
            %%%% corner to indiate that the user thinks it's the best
            %%%% fitting that can be done, but the fitting is still bad. 
            subplot(3,2,2);
            dum = axis;
            h=text(dum(1),dum(4),'bad');set(h,'color','r');
            plot(dum(2)/5,4*dum(4)/5,'r*');
            h=text(dum(2),dum(4),'InBad');set(h,'color','m');
            plot(4*dum(2)/5,4*dum(4)/5,'m*');
            disp('click on follicle wall or click twice to adjust threshold?');
            [xr,yr]= myginput(1,'arrow');
            if xr < dum(2)/5 & yr <dum(4)/5
                UserHappyWithResults = 0;
            elseif  xr > 4*dum(2)/5 & yr <dum(4)/5
                UserHappyWithResults = 0;
            elseif xr < dum(2)/5 & yr > 4*dum(4)/5  | isempty(EllipsesI) 
                UserHappyWithResults = 1;
                BadOuter = 1;
                BadInner = 1;
            elseif xr > 4*dum(2)/5 & yr > 4*dum(4)/5  | isempty(EllipsesI) 
                UserHappyWithResults = 1;
                BadOuter = 0;
                BadInner = 1;
            else
                UserHappyWithResults = 1;
                BadOuter = 0;
                BadInner = 0;
            end;
        end;   %%%%% this ends the loop on  UserHappyWithResults
        
        
        
        %%% Sometimes for bad fits all the variables except EllipsesO can
        %%% end up being empty,and EllipsesO still contains the raw points
        %%% For those cases just assign those
        %%% variables to equal EllipsesO
        %%% The first 100 points of InnerRawPoints and ensure follicle is
        %%% labeled bad.
        if exist('OuterRawPts') == 0
            OuterRawPts = [];
        end;
        if exist('InnerRawPts') == 0
            InnerRawPts = [];
        end;
        [yo, yoyo] = size(EllipsesO);
        if yo > yoyo,
            EllipsesO = EllipsesO';
        end;
        if length(EllipsesO) > 100
            % disp('eo'); disp(length(EllipsesO));
            EllipsesO = EllipsesO(:,1:100);BadOuter = 1;
        end;
        if isempty(EllipsesI)
            EllipsesI = EllipsesO; BadInner = 1; %  disp('ei'); 
        end;
        if isempty(OuterRawPts)
            OuterRawPts = EllipsesO;BadOuter = 1; %  disp('or'); 
        end;
        if isempty(InnerRawPts)
            InnerRawPts = EllipsesO;BadInner = 1; % disp('ir'); 
        end;
        
        [yo, yoyo] = size(OuterRawPts);
        if yo > yoyo,
            OuterRawPts = OuterRawPts';
        end;
        [yo, yoyo] = size(InnerRawPts);
        if yo > yoyo,
            InnerRawPts = InnerRawPts';
        end;
        
        % Put the follicle back in the right place in the original full
        % seciton image
        EllipsesO(1,:) = EllipsesO(1,:) + min(yINIT);
        EllipsesO(2,:) = EllipsesO(2,:) + min(xINIT);
        EllipsesI(1,:) = EllipsesI(1,:) + min(yINIT);
        EllipsesI(2,:) = EllipsesI(2,:) + min(xINIT);
        OuterRawPts(1,:) = OuterRawPts(1,:) +  min(yINIT);
        OuterRawPts(2,:) = OuterRawPts(2,:) +  min(xINIT);
        InnerRawPts(1,:) = InnerRawPts(1,:) +  min(yINIT);
        InnerRawPts(2,:) = InnerRawPts(2,:) +  min(xINIT);
        
        %%% plot the follicle in Figure 1 so user can watch progress.
        fig1;
        ho;
        plotv(EllipsesO(:,1:5:end),'y.');
        plotv(EllipsesI(:,1:5:end),'g.');
%         plotv(OuterRawPts,'c.');
%         plotv(InnerRawPts,'r.');


 
        if BadOuter == 1
            data = EllipsesO;
            h = text(mean(data(1,:))-textOffset,mean(data(2,:)),'BO'); set(h,'color','y');
            set(h,'color','y','FontSize',14);
        end;
        if BadInner == 1
            data = EllipsesO;
            h = text(mean(data(1,:)+textOffset),mean(data(2,:)),'BI'); set(h,'color','g');
            set(h,'color','g','FontSize',14);
        end;
        
        %%% Add to the list of fols2plot as well as BadOuter and BadInner list.
        fols2plot{length(fols2plot)+1} = EllipsesO;
        fols2plotI{length(fols2plotI)+1} = EllipsesI;
        BadOuters(length(BadOuters)+1) = BadOuter;
        BadInners(length(BadInners)+1)= BadInner;
        
        %%% Save results
        ORIGimgsize = size(abc);
        fname = ['Pad2_tif' int2str(tifNumber) '_fol' int2str2(folNumber,10) ];
        tempstring = ['save ' fname ' Ellipses* OuterRawPts InnerRawPts nFollicles  BadOuter BadInner ORIGimgsize'];
        disp(tempstring); eval(tempstring);
        
        folNumber = folNumber + 1;
        
    end % while loop on each of the follicles
    
    %%% at the start of the next section prompt the user to input
    %%% nFollicles in command window. 
    fig2;
    clf;text(1,1,'input N follicles in command window'); axx(-1,2);axy(-1,2);
    
end;


