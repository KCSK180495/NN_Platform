function run_bridge_checks(outputFolder)
% Run on MATLAB/Simulink; writes synthetic fixtures for the Python smoke check.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'nn_training_studio','matlab'));
if ~isfolder(outputFolder), mkdir(outputFolder); end
t = (0:0.001:1)';
a = timeseries(sin(2*pi*50*t), t, 'Name','Ia');
b = timeseries(cos(2*pi*50*t), t, 'Name','Ib');
a.DataInfo.Units = 'A'; b.DataInfo.Units = 'A';
ds = Simulink.SimulationData.Dataset;
ds = ds.addElement(a,'Ia'); ds = ds.addElement(b,'Ib');
[T,m] = nn_export_simulink(ds,fullfile(outputFolder,'normal.csv'),'Label','Normal','ModelName','synthetic_bridge_test');
assert(height(T)==1001 && isequal(T.Properties.VariableNames,{'Time_s','Ia','Ib','Label'}));
assert(strcmp(m.units.Ia,'A') && m.sample_time_s > 0);
nn_export_simulink(ds,fullfile(outputFolder,'normal.mat'),'Label','Normal');
b.Data = 1.2*b.Data;
fault = Simulink.SimulationData.Dataset;
fault = fault.addElement(a,'Ia'); fault = fault.addElement(b,'Ib');
nn_export_simulink(fault,fullfile(outputFolder,'fault.csv'),'Label','Fault');
nn_export_simulink(timeseries([a.Data b.Data],t,'Name','Current'),fullfile(outputFolder,'vector.mat'));
tableData = table(t,a.Data,b.Data,'VariableNames',{'Time_s','Ia','Ib'});
nn_export_simulink(tableData,fullfile(outputFolder,'table.csv'));
tt = timetable(seconds(t),a.Data,b.Data,'VariableNames',{'Ia','Ib'});
nn_export_simulink(tt,fullfile(outputFolder,'timetable.csv'));
logged = struct('time',t,'signals',struct('values',[a.Data b.Data],'label','Current'));
nn_export_simulink(logged,fullfile(outputFolder,'structure.mat'));
irregular = timeseries((1:4)',[0;.01;.035;.1],'Name','x');
expect_error(@()nn_export_simulink(irregular,fullfile(outputFolder,'invalid.csv')),'NNStudio:VariableStep');
nn_export_simulink(irregular,fullfile(outputFolder,'resampled.csv'),'SampleTime',.01);
mixed = Simulink.SimulationData.Dataset;
mixed = mixed.addElement(a,'Ia');
mixed = mixed.addElement(timeseries(b.Data(1:2:end),t(1:2:end)),'Ib');
expect_error(@()nn_export_simulink(mixed,fullfile(outputFolder,'invalid.csv')),'NNStudio:MultiRate');
nn_export_simulink(mixed,fullfile(outputFolder,'multirate.csv'),'SampleTime',.002,'Interpolation','previous');
original = fileread(fullfile(outputFolder,'normal.csv'));
expect_error(@()nn_export_simulink(ds,fullfile(outputFolder,'normal.csv')),'NNStudio:Exists');
assert(strcmp(original,fileread(fullfile(outputFolder,'normal.csv'))));
assert(~isfile(fullfile(outputFolder,'invalid.csv')));
% Real simulation with a temporary in-memory model, never a user's model.
[~,token] = fileparts(tempname);
model = ['nn_bridge_' token];
new_system(model);
cleanup = onCleanup(@()close_system(model,0));
add_block('simulink/Sources/Sine Wave',[model '/Sine'],'SampleTime','0.001');
add_block('simulink/Sinks/To Workspace',[model '/Log'],'VariableName','simdata','SaveFormat','Timeseries');
add_line(model,'Sine/1','Log/1');
set_param(model,'Solver','FixedStepDiscrete','FixedStep','0.001','StopTime','1');
[out, exported] = nn_run_and_export(model,fullfile(outputFolder,'simulated.csv'),'Variable','simdata','Label','Normal'); %#ok<ASGLU>
assert(height(exported)==1001);
fprintf('MATLAB bridge checks passed, including a real Simulink run.\n');
end

function expect_error(action, identifier)
try
    action();
catch err
    assert(strcmp(err.identifier,identifier),'Unexpected error: %s',err.message);
    return;
end
error('NNStudio:Test','Expected error %s was not raised.',identifier);
end
