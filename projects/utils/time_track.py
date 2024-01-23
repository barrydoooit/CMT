import json
import os
import time
import numpy as np



class TimeTracker:
    def __init__(self, output_file_path, parse_func=None, vis_func=None, is_tracking=False):
        self.timestamps = {}
        self.output_file_path = output_file_path
        self.parse_func = parse_func
        self.vis_func = vis_func
        self.is_tracking = is_tracking

    def register(self, name):
        if not self.is_tracking:
            return
        self.timestamps[name] = time.time()

    def get_time_diff(self, start_name, end_name):
        if start_name not in self.timestamps or end_name not in self.timestamps:
            raise ValueError(f"Timestamps for {start_name} or {end_name} not registered.")
        return self.timestamps[end_name] - self.timestamps[start_name]

    def reset(self):
        self.timestamps = {}

    def dump(self):
        if not self.is_tracking:
            return
        output = self.timestamps
        if self.parse_func is not None:
            output = self.parse_func(output)
        with open(self.output_file_path, 'a') as f:
            json.dump(output, f)
            f.write('\n')
        self.reset()

    def __del__(self):
        if not self.is_tracking:
            return
        vis_func = self.vis_func
        output_file_path = self.output_file_path
        if vis_func is not None and os.path.exists(output_file_path):
            vis_func(output_file_path)

class ModuleStartEndTimeTracker(TimeTracker):
    def __init__(self, output_file_path, is_tracking=False):
        def parse_func(timestamps):
            output = {}
            for module_name, module_timestamps in timestamps.items():
                # Store duration of each module
                output[module_name] = module_timestamps['end'] - module_timestamps['start']
            return output
        
        def vis_func(output_file_path):
            # read objects in json as dict
            with open(output_file_path, 'r') as f:
                lines = f.readlines()
                lines = [json.loads(line) for line in lines]
            # get all module names
            module_names = set()
            module_names.update(lines[0].keys())
            # get all time durations
            time_durations = {module_name: [] for module_name in module_names}
            for line in lines:
                for module_name, time_duration in line.items():
                    time_durations[module_name].append(time_duration)
            # Sort 
            # calculated mean and std of time durations
            time_durations_mean_std = {module_name: {} for module_name in module_names}
            for module_name, time_duration in time_durations.items():
                time_durations_mean_std[module_name]['mean'] = np.mean(time_duration)
                time_durations_mean_std[module_name]['std'] = np.std(time_duration)
                time_durations_mean_std[module_name]['median'] = np.median(time_duration)
                # First quater mean
                time_durations_mean_std[module_name]['q1-mean'] = np.mean(time_duration[:len(time_duration)//4])
                # First quater std
                time_durations_mean_std[module_name]['q1-std'] = np.std(time_duration[:len(time_duration)//4])
                # First quater median
                time_durations_mean_std[module_name]['q1-median'] = np.median(time_duration[:len(time_duration)//4])
                # Last quater mean
                time_durations_mean_std[module_name]['q4-mean'] = np.mean(time_duration[len(time_duration)//4*3:])
                # Last quater std
                time_durations_mean_std[module_name]['q4-std'] = np.std(time_duration[len(time_duration)//4*3:])
                # Last quater median
                time_durations_mean_std[module_name]['q4-median'] = np.median(time_duration[len(time_duration)//4*3:])
                # Record mean value without the small and large quarter of data


            # Print as 10 column table, times in miliseconds with 2 decimal places
            print(f"{'Module':<20}{'Mean':<10}{'Std':<10}{'Median':<10}{'Q1-Mean':<10}{'Q1-Std':<10}{'Q1-Median':<10}{'Q4-Mean':<10}{'Q4-Std':<10}{'Q4-Median':<10}\n")
            # Sort by module name
            time_durations_mean_std = dict(sorted(time_durations_mean_std.items(), key=lambda x: x[0]))
            for module_name, time_duration in time_durations_mean_std.items():
                # print mean, std, median, and also q1 q4 mean, std, median
                print(f"{module_name:<20} {time_duration['mean'] * 1000:<10.2f}"
                    f"{time_duration['std'] * 1000:<10.2f}"
                    f"{time_duration['median'] * 1000:<10.2f}"
                    f"{time_duration['q1-mean'] * 1000:<10.2f}"
                    f"{time_duration['q1-std'] * 1000:<10.2f}"
                    f"{time_duration['q1-median'] * 1000:<10.2f}"
                    f"{time_duration['q4-mean'] * 1000:<10.2f}"
                    f"{time_duration['q4-std'] * 1000:<10.2f}"
                    f"{time_duration['q4-median'] * 1000:<10.2f}")
            
            # save the above output to file
            with open(output_file_path.split('.')[0] + '.txt', 'w') as f:
                f.write(f"{'Module':<20}{'Mean':<10}{'Std':<10}{'Median':<10}{'Q1-Mean':<10}{'Q1-Std':<10}{'Q1-Median':<10}{'Q4-Mean':<10}{'Q4-Std':<10}{'Q4-Median':<10}\n")
                # print mean, std, median, and also q1 q4 mean, std, median
                for module_name, time_duration in time_durations_mean_std.items():
                    f.write(f"{module_name:<20} {time_duration['mean'] * 1000:<10.2f}"
                        f"{time_duration['std'] * 1000:<10.2f}"
                        f"{time_duration['median'] * 1000:<10.2f}"
                        f"{time_duration['q1-mean'] * 1000:<10.2f}"
                        f"{time_duration['q1-std'] * 1000:<10.2f}"
                        f"{time_duration['q1-median'] * 1000:<10.2f}"
                        f"{time_duration['q4-mean'] * 1000:<10.2f}"
                        f"{time_duration['q4-std'] * 1000:<10.2f}"
                        f"{time_duration['q4-median'] * 1000:<10.2f}\n")
               
        super().__init__(output_file_path, parse_func, vis_func, is_tracking)

    def register_start(self, module_name, ts=None):
        if not self.is_tracking:
            return
        module_time_dict = dict()
        module_time_dict['start'] = ts if ts is not None else time.time()
        self.timestamps[module_name] = module_time_dict
        return ts

    def register_end(self, module_name, ts=None):
        if not self.is_tracking:
            return
        if module_name not in self.timestamps:
            raise ValueError(f"Timestamps for {module_name} not registered.")
        self.timestamps[module_name]['end'] = ts if ts is not None else time.time()
        return ts
