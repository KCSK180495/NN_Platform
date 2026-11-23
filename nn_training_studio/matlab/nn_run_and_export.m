function [out, T, metadata] = nn_run_and_export(modelName, filename, varargin)
%NN_RUN_AND_EXPORT Run a user-configured model and export when simulation ends.
% nn_run_and_export('my_model','normal.csv','Label','Normal','SampleTime',1e-4)
% Enable signal logging first. Simulation errors never publish a dataset.
% Name/value options after filename are passed to nn_export_simulink.
out = sim(modelName, 'ReturnWorkspaceOutputs','on');
[T, metadata] = nn_export_simulink(out, filename, 'ModelName',modelName, varargin{:});
end
