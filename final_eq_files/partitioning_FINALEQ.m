load('FINAL_EQ.mat', 'PS');

load('FINAL_EQ.mat', 'well');

%here, we will split up the "PS" and "well" structures into manageable
%bites. This allows us to keep files on github, etc.

bins = ceil(linspace(1, 450, 10));

for i = 2:length(bins)
    tempPS = PS(1, bins(i-1):bins(i));
    tempwell = well(1, bins(i-1):bins(i));
    
    filename = sprintf('part_%d.mat', i-1);
    save(filename,'tempPS','tempwell')
end
%%
%now that we have split things up, we can put them back together.
load('part_1.mat', 'tempPS');
load('part_1.mat', 'tempwell');

PS = tempPS;
well = tempwell;

for i = 2:(length(bins)-1)
    filename = sprintf('part_%d.mat', i);
    load(filename, 'tempPS');
    load(filename, 'tempwell');
    PS = [PS tempPS];
    well = [well tempwell];
    
    clear tempPS tempwell
end


load('test.mat')
