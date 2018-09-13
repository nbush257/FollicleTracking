clear;
textOffset = 30;  % just for writing text on figure.
prefix = 'Pad2_'
tifNumbers = [6318:6320];

for qq = 1:length(tifNumbers)
    tifNumber = tifNumbers(qq)
    %%% Plot the original raw (color) image with each follicle over it.
    %%% Load and plot them one at a time so the user can fix them if
    %%% needed.  Once the user confirms the follicle is okay, re-save the
    %%% follicle file with the subscript _checked.
    %%% After all follicles have been confirmed, compile
    %%% all the follicles into a single section.  Create a directory with the name
    %%% of the section and put all the individual
    %%% follicle files into that directory.
    figure(1);clf;
    fname = ([prefix int2str(tifNumber) '.tif'])
    abc = imread(fname);
    imagesc(abc);ho;
    %%% Load up all the follicles that have been tracked but not checked
    s = dir(['*' int2str(tifNumber) '_fol*.mat' '']);
    maxfol = 0;
    if ~isempty(s)
        for ii = 1:length(s)
            clear Ellipse* Inner* Outer* Bad*
            data = s(ii).name;
            load(data);
            %%% plot in figure 1
            plotv(EllipsesO(:,1:end),'y.');
            plotv(InnerRawPts(:,1:end),'g.');
            if BadOuter == 1
                h = text(mean(EllipsesO(1,:))-textOffset,mean(EllipsesO(2,:)),'BO'); set(h,'color','y');
            end;
            if BadInner == 1
                h = text(mean(EllipsesO(1,:))+textOffset,mean(EllipsesO(2,:)),'BI'); set(h,'color','g');
            end;
            if BadInner == 0 & BadOuter == 0
                h = text(mean(EllipsesO(1,:))-textOffset,mean(EllipsesO(2,:)),'OK'); set(h,'color','k');
            end;
            %%% make a zoom plot in figure 2 for user to confirm.
            figure(2); clf;
            imagesc(abc);ho;
            plotv(EllipsesO(:,1:end),'y.');
            plotv(InnerRawPts(:,1:end),'g.');
            if BadOuter ==1
                h = text(max(EllipsesO(1,:)),mean(EllipsesO(2,:)),'BadOuter'); set(h,'color','y','FontSize',14);
            elseif BadOuter == 0
                h = text(max(EllipsesO(1,:)),mean(EllipsesO(2,:)),'GoodOuter'); set(h,'color','y','FontSize',14);
            end;
            if BadInner ==1
                h = text(max(EllipsesO(1,:)),mean(EllipsesO(2,:))+textOffset,'BadInner'); set(h,'color','g','FontSize',14);
            elseif BadInner == 0
                h = text(max(EllipsesO(1,:)),mean(EllipsesO(2,:))+textOffset,'GoodInner'); set(h,'color','g','FontSize',14);
            end;
            sf = 1.1;
            minx = min(EllipsesO(1,:))/sf; maxx = max(EllipsesO(1,:))*sf;
            miny = min(EllipsesO(2,:))/sf; maxy = max(EllipsesO(2,:))*sf;
            axx(minx,maxx); axy(miny,maxy);
            h=text(minx + 0.1*(maxx-minx), miny+ 0.1*(maxx-minx), 'OK as labeled'); set(h,'color','y','FontSize',14);
            h=text(minx + 6*(maxx-minx)/10, miny+ 0.1*(maxx-minx), 'Both Good'); set(h,'color','y','FontSize',14);
            h=text(minx+(maxx-minx)/10, miny + 9*(maxy-miny)/10, 'Both bad'); set(h,'color','r','FontSize',14);
            h=text(minx + 6*(maxx-minx)/10, miny + 9*(maxy-miny)/10, 'Only Inner bad');
            set(h,'color','m','FontSize',14);
            vline(maxy,maxy*0.97,mean([minx,maxx]),'w');
            vline(miny,miny*1.05,mean([minx,maxx]),'w');
            hline(minx,minx*1.05,mean([miny,maxy]),'w');
            hline(maxx*0.97,maxx,mean([miny,maxy]),'w');
            title(['Section ' int2str(tifNumber)]);
            [xr,yr]= myginput(1,'arrow');
            if yr < mean([miny,maxy]) & xr < mean([minx,maxx]) % upper left
                % follicle correctly labeled -- change nothing
            elseif yr < mean([miny,maxy]) & xr > mean([minx,maxx]) % upper right -- both are good
                %%% but the user accidentally labeled one or more as bad
                BadOuter = 0;
                BadInner = 0;
            elseif yr > mean([miny,maxy]) & xr < mean([minx,maxx])  % lower left both bad
                BadOuter = 1;
                BadInner = 1;
            elseif yr > mean([miny,maxy]) & xr > mean([minx,maxx])  % lower right only inner bad
                BadOuter = 0;
                BadInner = 1;
            end;
            
            fig1;
            if BadInner == 0 & BadOuter == 0
                h = text(mean(EllipsesO(1,:))-textOffset,min(EllipsesO(2,:)),'OK'); set(h,'color','c','FontSize',14);
            end;
            if BadOuter == 1
                h = text(min(EllipsesO(1,:))-textOffset,min(EllipsesO(2,:))-textOffset,'BO'); set(h,'color','y','FontSize',14);
            end;
            if BadInner == 1
                h = text(min(EllipsesO(1,:))+textOffset,min(EllipsesO(2,:))-textOffset,'BI'); set(h,'color','g','FontSize',14);
            end;
            
            mf = eval(data(end-5:end-4));
            if mf >maxfol
                maxfol = mf;
            end;
            
            dataSaveName = [data(1:end-10),'_checked' data(end-9:end)];
            
            tempstring = ['save ' dataSaveName ' EllipsesO EllipsesI OuterRawPts InnerRawPts nFollicles  BadOuter BadInner'];
            disp(tempstring); eval(tempstring);
            
        end; %%% Loop on follicle number
    else
        maxfol = 0;
    end;   %%% If there were any tracked follicles
    
    %%% This if statement is a paranoid sanity check
    %%% They should always be equal
    if maxfol ~= length(s)
        disp('error in fol number');
    end;
    
    disp('Press any key if everything looks ok');
    pause;
    
end %%  Loop on tif numbers


