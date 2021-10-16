infolder='/path/to/wrl/folder/';
outfolder='/path/to/input/data/folder/all';
%keep that '/all'
dire=dir([infolder '*.wrl']);
for i=1:length(dire)
    cfname=dire(i).name;
    if cfname(end-4)=='l'
        continue
    end
    abf=cfname;
    agf=cfname;
    agf(end-4)='l';
%     producepointdata(infolder,abf,infolder,agf,outfolder);
    producepointdataall2(infolder,abf,infolder,agf,outfolder);
end
