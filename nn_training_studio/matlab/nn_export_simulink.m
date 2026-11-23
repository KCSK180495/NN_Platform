function [T, metadata] = nn_export_simulink(data, filename, varargin)
%NN_EXPORT_SIMULINK Export logged signals for NN Training Studio.
% out = sim('my_model');
% nn_export_simulink(out,'normal.csv','Label','Normal');
% nn_export_simulink(out,'run.mat','SampleTime',1e-4);
% Accepts SimulationOutput, Dataset, timeseries, duration-row timetable,
% numeric table, Structure with time, or [time signals] numeric matrix.
% Multirate/variable-step results require an explicit SampleTime. No MATLAB
% objects are saved: MAT output uses plain numeric arrays and -v7 format.

p = inputParser;
addParameter(p, 'Variable', 'logsout', @(x) ischar(x) || isstring(x));
addParameter(p, 'Label', []);
addParameter(p, 'SampleTime', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && isfinite(x) && x > 0));
addParameter(p, 'Interpolation', 'linear', @(x) any(strcmp(x, {'linear','previous','nearest'})));
addParameter(p, 'ModelName', '', @(x) ischar(x) || isstring(x));
parse(p, varargin{:});
options = p.Results;
if isa(data, 'Simulink.SimulationOutput')
    if ~isempty(data.ErrorMessage)
        error('NNStudio:SimulationFailed', '%s', data.ErrorMessage);
    end
    variables = who(data);
    selected = char(options.Variable);
    if ~any(strcmp(variables, selected))
        if strcmp(selected, 'logsout') && any(strcmp(variables, 'yout'))
            selected = 'yout';
        else
            error('NNStudio:Logging', 'No %s output. Enable signal logging or pass ''Variable'',''your_workspace_variable''.', selected);
        end
    end
    data = data.get(selected);
end
series = collect_signals(data, '');
if isempty(series)
    error('NNStudio:Empty', 'No numeric logged signals found. Enable signal logging first.');
end
for k = 1:numel(series)
    t = series(k).time;
    if numel(t) < 2 || any(~isfinite(t)) || any(diff(t) <= 0)
        error('NNStudio:Time', 'Signal %s needs at least two strictly increasing finite time points.', series(k).name);
    end
    if any(~isfinite(series(k).values(:)))
        error('NNStudio:Values', 'Signal %s contains nonfinite samples.', series(k).name);
    end
end
time = series(1).time;
if isempty(options.SampleTime)
    for k = 2:numel(series)
        if ~isequal(time, series(k).time)
            error('NNStudio:MultiRate', 'Signals use different time grids. Specify ''SampleTime'',Ts and an interpolation method.');
        end
    end
    dt = diff(time);
    Ts = median(dt);
    if any(abs(dt-Ts) > max(1e-12, abs(Ts)*1e-5))
        error('NNStudio:VariableStep', 'Variable-step logging needs ''SampleTime'',Ts for a uniform export.');
    end
else
    Ts = options.SampleTime;
    first = max(arrayfun(@(s) s.time(1), series));
    last = min(arrayfun(@(s) s.time(end), series));
    count = floor((last-first)/Ts) + 1;
    if count < 2 || count > 2000000
        error('NNStudio:Size', 'Choose a SampleTime giving 2 to 2,000,000 samples in the common signal interval.');
    end
    time = first + (0:count-1)' * Ts;
end
if numel(time) > 2000000
    error('NNStudio:Size', 'Export supports at most 2,000,000 rows. Crop the run or increase SampleTime.');
end
signals = zeros(numel(time), numel(series));
for k = 1:numel(series)
    if isempty(options.SampleTime)
        signals(:,k) = series(k).values;
    else
        signals(:,k) = interp1(series(k).time, series(k).values, time, options.Interpolation);
    end
end
signal_names = matlab.lang.makeValidName({series.name});
signal_names = matlab.lang.makeUniqueStrings(signal_names, ...
    {'Time_s','Label','simulink_run_id','simulink_source_row','annotation_segment_id'});
T = array2table([time signals], 'VariableNames', [{'Time_s'} signal_names]);
hasLabel = ~isempty(options.Label);
if hasLabel
    if isnumeric(options.Label) || islogical(options.Label)
        validateattributes(options.Label, {'numeric','logical'}, {'scalar','real','finite'});
        labels = repmat(options.Label, numel(time), 1);
    else
        label = string(options.Label);
        if ~isscalar(label) || ismissing(label) || strlength(strtrim(label)) == 0
            error('NNStudio:Label', 'Label must be one nonempty confirmed label per run.');
        end
        labels = cellstr(repmat(label, numel(time), 1));
    end
    T.Label = labels;
end
units = struct();
for k = 1:numel(series)
    units.(signal_names{k}) = series(k).unit;
end
[~, runId] = fileparts(tempname);
metadata = struct('format','nnstudio-simulink', 'version',1, ...
    'model_name',char(options.ModelName), 'run_id',runId, ...
    'sample_time_s',Ts, 'signal_names',{signal_names}, 'units',units, ...
    'interpolation',char(options.Interpolation), ...
    'created_utc',char(datetime('now','TimeZone','UTC','Format','yyyy-MM-dd''T''HH:mm:ss''Z''')));
if hasLabel
    metadata.target_column = 'Label';
end
metadata_json = jsonencode(metadata);
[folder, base, extension] = fileparts(char(filename));
if isempty(folder), folder = pwd; end
if ~any(strcmpi(extension, {'.csv','.mat'}))
    error('NNStudio:Format', 'Output filename must end with .csv or .mat.');
end
if ~isfolder(folder)
    error('NNStudio:Folder', 'Output folder does not exist: %s', folder);
end
destination = fullfile(folder, [base extension]);
sidecar = fullfile(folder, [base '.simulation.json']);
if isfile(destination) || (strcmpi(extension,'.csv') && isfile(sidecar))
    error('NNStudio:Exists', 'Export already exists. Choose a new filename for each simulation run.');
end
staging = [tempname(folder) extension];
stagingJson = [tempname(folder) '.json'];
cleanup = onCleanup(@() remove_staging(staging, stagingJson));
if strcmpi(extension, '.csv')
    writetable(T, staging, 'FileType','text', 'Delimiter',',', 'Encoding','UTF-8');
    fid = fopen(stagingJson, 'w', 'n', 'UTF-8');
    if fid < 0, error('NNStudio:Write', 'Cannot create export metadata.'); end
    closeFile = onCleanup(@() fclose(fid));
    fprintf(fid, '%s', metadata_json);
    clear closeFile;
    movefile(stagingJson, sidecar);
else
    if hasLabel
        save(staging, 'time','signals','signal_names','labels','metadata_json','-v7');
    else
        save(staging, 'time','signals','signal_names','metadata_json','-v7');
    end
end
movefile(staging, destination);
fprintf('NN Studio export: %s (%d samples, %.9g Hz)\n', destination, height(T), 1/Ts);
end

function series = collect_signals(data, prefix)
series = struct('name',{}, 'time',{}, 'values',{}, 'unit',{});
if isa(data, 'Simulink.SimulationData.Dataset')
    for k = 1:data.numElements
        element = data.getElement(k);
        name = element.Name;
        if isempty(name), name = sprintf('Signal_%d', k); end
        series = [series collect_signals(element.Values, join_name(prefix, name))]; %#ok<AGROW>
    end
elseif isa(data, 'timeseries')
    values = data.Data;
    if ~data.IsTimeFirst
        values = permute(values, [ndims(values) 1:ndims(values)-1]);
    end
    if isempty(prefix), prefix = data.Name; end
    if isempty(prefix), prefix = 'Signal'; end
    scale = time_scale(data.TimeInfo.Units);
    series = numeric_series(double(data.Time(:))*scale, values, prefix, char(string(data.DataInfo.Units)));
elseif istimetable(data)
    if ~isduration(data.Properties.RowTimes)
        error('NNStudio:Time', 'Timetable row times must be durations in simulation time.');
    end
    t = seconds(data.Properties.RowTimes);
    for k = 1:width(data)
        name = data.Properties.VariableNames{k};
        unit = data.Properties.VariableUnits{k};
        series = [series numeric_series(t, data.(name), join_name(prefix,name),unit)]; %#ok<AGROW>
    end
elseif istable(data)
    names = data.Properties.VariableNames;
    index = find(ismember(lower(string(names)), ["time_s","time","tout","t"]),1);
    if isempty(index), error('NNStudio:Time', 'Table needs a numeric time column in seconds.'); end
    t = data.(names{index});
    for k = setdiff(1:width(data), index, 'stable')
        series = [series numeric_series(t,data.(names{k}),join_name(prefix,names{k}),data.Properties.VariableUnits{k})]; %#ok<AGROW>
    end
elseif isstruct(data) && isscalar(data) && isfield(data,'time') && isfield(data,'signals')
    for k = 1:numel(data.signals)
        name = sprintf('Signal_%d',k);
        if isfield(data.signals(k),'label') && ~isempty(data.signals(k).label)
            name = data.signals(k).label;
        end
        series = [series numeric_series(data.time,data.signals(k).values,join_name(prefix,name),'')]; %#ok<AGROW>
    end
elseif isstruct(data) && isscalar(data)
    names = fieldnames(data);
    for k = 1:numel(names)
        series = [series collect_signals(data.(names{k}),join_name(prefix,names{k}))]; %#ok<AGROW>
    end
elseif isnumeric(data) && ismatrix(data) && size(data,2) >= 2
    series = numeric_series(data(:,1),data(:,2:end),join_name(prefix,'Signal'),'');
else
    error('NNStudio:Type', 'Unsupported logged data at %s. Export real numeric timeseries, timetables or Structure with time.', prefix);
end
end

function series = numeric_series(time, values, name, unit)
if ~(isnumeric(values) || islogical(values)) || ~isreal(values) || ~isnumeric(time) || ~isreal(time)
    error('NNStudio:Numeric', 'Signal %s and its time must be real numeric values.', name);
end
time = double(time(:));
if size(values,1) ~= numel(time)
    error('NNStudio:Shape', 'Signal %s must have samples along its first dimension.',name);
end
values = reshape(double(values),numel(time),[]);
series = struct('name',{}, 'time',{}, 'values',{}, 'unit',{});
for k = 1:size(values,2)
    component = name;
    if size(values,2) > 1, component = sprintf('%s_%d',name,k); end
    series(end+1) = struct('name',component,'time',time,'values',values(:,k),'unit',unit); %#ok<AGROW>
end
end

function name = join_name(prefix, suffix)
if isempty(prefix), name = char(suffix); else, name = [char(prefix) '_' char(suffix)]; end
end

function scale = time_scale(unit)
switch lower(char(unit))
    case {'seconds','second','sec','s'}, scale = 1;
    case {'milliseconds','millisecond','ms'}, scale = 1e-3;
    case {'microseconds','microsecond','us'}, scale = 1e-6;
    case {'minutes','minute'}, scale = 60;
    case {'hours','hour'}, scale = 3600;
    case {'days','day'}, scale = 86400;
    otherwise, error('NNStudio:Units','Unsupported timeseries time unit: %s',char(unit));
end
end

function remove_staging(varargin)
for k = 1:nargin
    if isfile(varargin{k}), delete(varargin{k}); end
end
end
