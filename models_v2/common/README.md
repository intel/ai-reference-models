# Overview

[benchmark.py]: benchmark.py
[telemetry.py]: telemetry.py
[js_merge.py]: js_merge.py
[js_sysinfo.py]: js_sysinfo.py
[json_to_csv.py]: json_to_csv.py
[parse_telemetry.py]: parse_telemetry.py
[parse_result.py]: parse_result.py
[save_to_json.py]: save_to_json.py

[DKMS]: https://github.com/dell/dkms
[Docker]: https://www.docker.com/
[lshw]: https://github.com/lyonel/lshw
[svr-info]: https://github.com/intel/svr-info
[PyTorch]: https://pytorch.org/

[xpu-smi]: https://intel.github.io/xpumanager/smi_install_guide.html
[nvidia-smi]: https://developer.nvidia.com/nvidia-system-management-interface

This folder contains common Python utilities and modules which can be reused across project.

| Utility              | Description                                              |
| ---------------------| -------------------------------------------------------- |
| [benchmark.py]       | Tool to benchmark application given in cmdline arguments |
| [telemetry.py]       | Tool to capture GPU telemetry                            |
| [js_merge.py]        | Tool to merge few JSON files together                    |
| [js_sysinfo.py]      | Tool to dump system information in JSON format           |
| [json_to_csv.py]     | Tool to dump few json files into single CSV              |
| [parse_telemetry.py] | Summarize SMI output from CSV to JSON                    |
| [parse_result.py]    | Sample results parser (view file for cmdline options)    |
| [save_to_json.py]    | Tool to save/amend data to json file                     |

# benchmark.py

> [!NOTE]
> Tool requires elevated privileges (root).

Usage:
```
benchmark [-h] [--indent INDENT] [--socket SOCKET] [--telemetry] [--metadata METADATA] --output_dir OUTPUT_DIR --platform PLATFORM --profile PROFILE app-cmdline
```

Tool benchmarks application given on the command line with `app-cmdline` arguments. `app-cmdline` is considered by the tool as a template with parameters specified with `{param}` notation. Tool performs parameters substitution at runtime taking parameter values from CSV profile given with `--profile` argument or from the predefined parameters list (see table below). Thus, tool executes as many benchmark tests as defined in the CSV profile.

Tool generates outputs in the directory given with `--output_dir` option. Outputs layout is as follows:
```
.
├── profile.csv               # copy of profile
├── sysinfo.json              # baremetal sysinfo description
├── test_{i}                  # output directory of i-th test
│   ├── test.csv              # test csv definition (i-th line from profile.csv)
│   ├── results.json          # results output from the test complying with the schema
│   ├── *_smi_dump.csv        # raw SMI output
│   ├── telemetry.json        # processed telemetry stats from SMI
│   └── *                     # whatever other output files test produces
├── results_test_{i}.json     # ultimate report for i-th test
└── summary.csv               # engineering summary for debug
```

Note that each test execution of `app-cmdline` must produce outputs to `test_{i}` folder and one of the outputs must be `results.json` file. Use predefined `output_dir` parameter on `app-cmdline` to specify path to `test_{i}` output directory (don't confuse this parameter with `--output_dir` option).

During the run tool collects system information - see `js_sysinfo.py` - and optionally collects GPU telemetry - see `telemetry.py` - and amends it to each test results description. Tool requires elevated privileges.

**Example:**

```
# echo "os,arch" > profile.csv
# echo "Linux,x86_64" >> profile.csv

# export RESULTS='{{ \"os\": \"{os}\", \"arch\": \"{arch}\" }}'
# sudo PYTHONPATH=./models_v2/common \
    python3 -m benchmark --output_dir /tmp/output --profile profile.csv --indent 4 --telemetry --socket /tmp/telemetry.s \
      /bin/bash -c "echo $RESULTS >{output_dir}/results.json"
```

This example benchmarks `/bin/bash` executing `echo` command which outputs to `results.json` required by the `benchmark.py` script. Note usage of environment variable `$RESULTS` in the `app-cmdline` as well as `{os}` and `{arch}` template parameters being taken from `profile.csv`. Mind that double curvy brackets `{{ }}` is a way to escape curvy brackets for Python format function.

See below snapshot of input CSV profile and results produced by the run.

```
# cat profile.csv
os,arch
Linux,x86_64

# tree /tmp/output/
/tmp/output/
├── profile.csv
├── results_test_1.json
├── sysinfo.json
└── test_1
    ├── test.csv
    ├── results.json
    ├── *_smi_dump.csv
│   ├── telemetry.json
    └── *

1 directory, 5 files

# cat /tmp/output/test_1/results.json
{ "os": "Linux", "arch": "x86_64" }

```

## Predefined template parameters

| Parameter      | Description                                       |
| -------------- | ------------------------------------------------- |
| `{output_dir}` | Directory where `app-cmdline` should write output |
| `{socket}`     | Socket to control telemetry capture               |

## Options

| Option                        |                                                            |
| ----------------------------- | ---------------------------------------------------------- |
| `-h`, `--help`                | Show this help message and exit                            |
| `--indent INDENT`             | Indent for json.dump()                                     |
| `--output_dir OUTPUT_DIR`     | Path to store outputs                                      |
| `--profile PROFILE`           | Profile with tests list                                    |
| `--platform PLATFORM`         | GPU platform benchmark is being ran on                     |
|                               | Options are: `Flex`, `Max`, `CUDA`                         |
| `--socket SOCKET`             | Socket to control telemetry capture                        |
|                               | This argument is optional                                  |
| `--telemetry`                 | Enable GPU telemetry capture during benchmarking           |
| `--metadata`                  | Space separated key=value pairs to amend to json reports   |

# telemetry.py

> [!NOTE]
> Tool requires elevated privileges (root).

Usage:
```
telemetry [-h] [--socket SOCKET] --output_dir OUTPUT_DIR  --platform Flex
```

Tool starts up GPU telemetry capture using the platform specific SMI tool - see installation instructions for [xpu-smi] on `Flex`/`Max` GPUs and [nvidia-smi] for `CUDA` GPUs.

Raw data is written to a CSV file in the specified ``--output_dir`` (file name depends on underlying GPU type, NVidia or Intel).

Tool behavior can be controlled by sending the following messages to the UNIX socket specified via `--socket` option:
* `start` - start telemetry capture
* `stop` - stop telemetry capture
* `kill` - kill telemetry process

If socket is not specified, tool starts telemetry capture immediately till it's terminated by CTRL+C.

For convenience tool wraps actions over socket in utility functions and exposes them on command line via `--start`, `--stop` and `--kill` options.

**Example:**

No `--socket`:
```sh
# sudo python telemetry.py --platform Flex --output_dir /tmp/output/ &

# ls /tmp/output/
xpu_smi_dump.csv

# cat /tmp/output/xpu_smi_dump.csv
Timestamp    ,DeviceId ,GPU Utilization (%) ,GPU Power (W) ,GPU Frequency (MHz) , ....
03:06:02.000 ,       0 ,              30.09 ,        57.38 ,               2050 , ....
03:06:03.000 ,       0 ,              98.30 ,        70.77 ,               1950 , ....
03:06:04.000 ,       0 ,              98.30 ,        70.80 ,               2050 , ....
03:06:05.000 ,       0 ,              98.29 ,        70.81 ,               2050 , ....
03:06:06.000 ,       0 ,              98.29 ,        70.90 ,               1950 , ....
03:06:07.000 ,       0 ,              98.24 ,        70.80 ,               1950 , ....
03:06:08.000 ,       0 ,              98.26 ,        70.88 ,               2050 , ....
03:06:09.000 ,       0 ,              98.28 ,        70.96 ,               1950 , ....
....

# sudo pkill -f telemetry
```

With ``--socket``:
```sh
# sudo python3 telemetry.py --platform Flex --output_dir /tmp/output/ --socket /tmp/telemetry.s &
# [telemetry.py: Listens for messages over the socket]

# ls /tmp/output/

# python3 telemetry.py --socket /tmp/telemetry.s --start

# [Some app is running for which telemetry is collected]

# python3 telemetry.py --socket /tmp/telemetry.s --stop
# python3 telemetry.py --socket /tmp/telemetry.s --kill

# ls /tmp/output/
xpu_smi_dump.csv

# cat /tmp/output/xpu_smi_dump.csv
Timestamp    ,DeviceId ,GPU Utilization (%) ,GPU Power (W) ,GPU Frequency (MHz) , ....
03:06:02.000 ,       0 ,              30.09 ,        57.38 ,               2050 , ....
03:06:03.000 ,       0 ,              98.30 ,        70.77 ,               1950 , ....
03:06:04.000 ,       0 ,              98.30 ,        70.80 ,               2050 , ....
03:06:05.000 ,       0 ,              98.29 ,        70.81 ,               2050 , ....
03:06:06.000 ,       0 ,              98.29 ,        70.90 ,               1950 , ....
03:06:07.000 ,       0 ,              98.24 ,        70.80 ,               1950 , ....
03:06:08.000 ,       0 ,              98.26 ,        70.88 ,               2050 , ....
03:06:09.000 ,       0 ,              98.28 ,        70.96 ,               1950 , ....
....

```

## Options

| Option                |                                                    |
| --------------------- | -------------------------------------------------- |
| `-h`, `--help`        | Show this help message and exit                    |
| `--indent INDENT`     | Indent for json.dump()                             |
| `--output_dir`        | Directory where raw telemetry should be written to |
| `--platform PLATFORM` | GPU platform telemetry is being collected on       |
|                       | Options are: `Flex`, `Max`, `CUDA`                 |
| `--socket SOCKET`     | Socket to control telemetry capture                |

# js_merge.py

Usage:
```
js_merge [-h] [--indent INDENT] file [file ...]
```

Tool merges few JSON files together preserving all unique values. Values type mismatch is considered a fatal error. With few values of the same type for the same key only one of the first input is kept with the warning printed out.

Positional arguments:
| Argument | Description        |
| -------- | ------------------ |
| file     | JSON file to merge |

Options:
| Option            |                                 |
| ----------------- | ------------------------------- |
| `-h`, `--help`    | Show this help message and exit |
| `--indent INDENT` | Indent for json.dump()          |

# js_sysinfo.py

> [!NOTE]
> Running with elevated privileges (root) recommended.

Usage:
```
js_sysinfo [-h] [--verbose] [-a] [--dkms] [--docker] [--dpkg] [--pytorch] [--indent INDENT]
```

Tool dumps system information in JSON format. By default minimal information is printed including underlying OS, kernel and python versions. If tool is running on baremetal, additionally key hardware information will be printed. To extend information collected by the tool, use command line options matching info you want to collect or use `-a` (`--all`) option to collect as much info as possible. Note that using these options requires additional tools to be available on the system such as [lshw], [svr-info], [DKMS], etc. If additional tool is missing, `js_sysinfo` script will skip collection with this tool and still produce an output. Add `--verbose` to command line to check for errors. Note also that some of the additional tools require or collect much more information if executed with elevated privileges (under root). Thus, to collect as much info as possible, execute `js_sysinfo.py` with elevated privileges as well:

```
sudo PATH=/path/to/tools:$PATH python3 js_sysinfo.py --all --verbose --indent 4 --output sysinfo.json
```

Options:
| Option            | Description                             |
| ----------------- | --------------------------------------- |
| `-h`, `--help`    | Show this help message and exit         |
| `-v`, `--verbose` | Enable verbose output                   |
| `-q`, `--quiet`   | Be quiet and suppress message to stdout |

Collect information options:
| Option            | Description                                    |
| ----------------- | ---------------------------------------------- |
| `-a, --all`       | Collect information for all known components   |
| `--dkms`          | Collect information for [DKMS]                 |
| `--docker`        | Collect information for [Docker]               |
| `--dpkg`          | Collect information for some key DPKG packages |
| `--lshw`          | Collect information with [lshw]                |
| `--pytorch`       | Collect information for [PyTorch]              |
| `--svrinfo`       | Collect information with [svr-info] (should be in PATH) |

JSON dump options:
| Option                | Decsription                     |
| --------------------- | ------------------------------- |
| `-o, --output OUTPUT` | File to store output            |
| `--indent INDENT`     | Indent for json.dump()          |

# json_to_csv.py

Usage:
```
json_to_csv [-h] -o OUTPUT file [file ...]
```

Tool dumps few JSON files into a single CSV file. Arrays and nested JSON keys are handled by serializing through joining indexes or keys with the `.` delimiter.

**Example:**

```
# cat a.json
{
    "os": "Linux",
    "arch": "x86_64",
    "software": [ "lshw", "dmidecode" ]
}

# cat b.json
{
    "os": "Linux",
    "arch": "x86",
    "software": [ "lshw", "svr-info" ]
}

# python3 ./models_v2/common/json_to_csv.py -o c.csv a.json b.json

# cat c.csv
os,arch,software.0,software.1
Linux,x86_64,lshw,dmidecode
Linux,x86,lshw,svr-info

```

Positional arguments:
| Argument | Description       |
| -------- | ----------------- |
| file     | JSON file to dump |

Options:
| Option                |                                 |
| --------------------- | ------------------------------- |
| `-h`, `--help`        | Show this help message and exit |
| `-o, --output OUTPUT` | File to store output            |

# parse_telemetry.py

Usage:
```
parse_telemetry [-h] [--indent INDENT] --output OUTPUT --input INPUT --platform PLATFORM
```

SMI tool ([xpu-smi], [nvidia-smi], and [telemetry.py]) produces a CSV with raw telemetry stats collected with a certain time interval.
This tool computes the min, max, mean, and stdev for a selection of collected telemetry stats and appends them to the requested JSON file.
This is used to incorporate the summarized telemetry stats in a condensed format into a JSON report.

Example usage:
```sh
# cat telemetry/xpu_smi_dump.csv
GPU Utilization (%),GPU Temperature (C)
0,40
50,50
100,60

# cat results/results.json
{
    "results": {
        "metrics": {
            "throughput" {
                "min": 100.0,
                "max": 110.0,
                "avg": 105.0,
                "stdev": 2.0,
                "unit": "images/s"
            }
        }
    }
}

# python parse_telemetry.py --input telemetry --output results/results.json --platform Flex

# cat results/results.json
{
    "results": {
        "metrics": {
            "throughput" {
                "min": 100.0,
                "max": 110.0,
                "avg": 105.0,
                "stdev": 2.0,
                "unit": "images/s"
            },
            "gpu-utilization" {
                "min": 0.0,
                "max": 50.0,
                "avg": 100.0,
                "stdev": 40.825.0,
                "unit": "%"
            },
            "gpu-temperature" {
                "min": 40.0,
                "max": 60.0,
                "avg": 50.0,
                "stdev": 8.165,
                "unit": "C"
            }
        }
    }
}
```

## Options

| Option                |                                          |
| --------------------- | ---------------------------------------- |
| `-h`, `--help`        | Show this help message and exit          |
| `--output OUTPUT`     | File to append telemetry data to         |
| `--input INPUT`       | Path to CSV file with SMI telemetry data |
| `--platform PLATFORM` | `Flex`, `Max`, `CUDA`                    |

# save_to_json.py

Usage:
```
save_to_json [-h] [--amend] [--indent INDENT] file key=value [key=value ...]
```

Tool saves given key/value pairs to the JSON file. Nested keys should be delimited by `.` (period). With `--amend` option keys are amended to the output file. Amended keys take precedence over existing keys in output file.

Options:
| Option                |                                 |
| --------------------- | ------------------------------- |
| `-h`, `--help`        | Show this help message and exit |
| `--amend`             | Amend data to output file       |
| `--indent INDENT`     | Indent for json.dump()          |

Positional arguments:
| Argument    | Description                                               |
| ----------- | --------------------------------------------------------- |
| `file`      | JSON file to store output                                 |
| `key=value` | key/value pair to amend (`key.key=value` for nested keys) |
