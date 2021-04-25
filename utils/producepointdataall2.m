function [ output_args ] = producepointdataall2( abinfolder,abfile,aginfolder,agfile,outfolder)
%UNTITLED2 Summary of this function goes here
threshold1=2;

pof=[outfolder '/points/'];
lof=[outfolder '/points_label/'];
[vb,~,cb,nb]=read_wrl2([abinfolder abfile]);
cb=cb';
nb=nb';
[vg,~,cg,ng]=read_wrl2([aginfolder agfile]);
cg=cg';
ng=ng';
vecb=unique(vb','rows');
vecg=unique(vg','rows');

ib=ismember(vecb,vb','rows');
ig=ismember(vecg,vg','rows');

colb=cb(ib,:);
colg=cg(ig,:);

nomb=nb(ib,:);
nomg=ng(ig,:);

cbg1=ismembertol(vecb, vecg, threshold1, 'ByRows', true, 'OutputAllIndices', true, 'DataScale', [1,1,1]);
cgb1=ismembertol(vecg, vecb, threshold1, 'ByRows', true, 'OutputAllIndices', true, 'DataScale', [1,1,1]);
% iob=find(cbg1==1);
% iog=find(cgb1==1);
% nvecb=vecb(iob,:);
% nvecg=vecg(iog,:);

% cbg2=ismembertol(nvecb, nvecg, threshold1, 'ByRows', true, 'OutputAllIndices', true, 'DataScale', [1,1,1]);
% cgb2=ismembertol(nvecg, nvecb, threshold1, 'ByRows', true, 'OutputAllIndices', true, 'DataScale', [1,1,1]);

% length(nvecg)
% length(nvecb)
% mkdir(outfolder)
mkdir(pof)
mkdir(lof)
% 
% dlmwrite([pof agfile(1:end-3) 'pts'],nvecg,'delimiter',' ');
% dlmwrite([pof abfile(1:end-3) 'pts'],nvecb,'delimiter',' ');
% csvwrite([lof agfile(1:end-3) 'seg'],cgb2+1);
% csvwrite([lof abfile(1:end-3) 'seg'],cbg2+1);

dlmwrite([pof agfile(1:end-3) 'pts'],[vecg nomg colg],'delimiter',' ');
dlmwrite([pof abfile(1:end-3) 'pts'],[vecb nomb colb],'delimiter',' ');
csvwrite([lof agfile(1:end-3) 'seg'],cgb1+1);
csvwrite([lof abfile(1:end-3) 'seg'],cbg1+1);

end

