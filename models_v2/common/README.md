# Overview

This folder contains common Python utilities and modules which can be reused across project.

[js_merge.py]: js_merge.py
[js_sysinfo.py]: js_sysinfo.py
[parse_result.py]: parse_result.py

[DKMS]: https://github.com/dell/dkms
[Docker]: https://www.docker.com/
[lshw]: https://github.com/lyonel/lshw
[svr-info]: https://github.com/intel/svr-info
[PyTorch]: https://pytorch.org/

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
