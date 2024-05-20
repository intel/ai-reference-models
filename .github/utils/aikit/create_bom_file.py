# input - drop_delivery_dir, boms_dir, component
# output - $bom_dir/l_<$component>.txt

import argparse
import os
import subprocess


def get_cksum(fname):
    result = subprocess.run(['cksum', fname], stdout=subprocess.PIPE)
    # e.g output: <cksum> <size> <filename>
    # 274833834 11348 /Users/jpatil/dev/drop_1.8.0/models/LICENSE
    return result.stdout.decode('UTF-8').split(' ')[0]

def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

def generate_bom_file(bom_file_name, drop_delivery_dir, component):
    # Remove existing bom file
    if os.path.isfile(bom_file_name):
        os.remove(bom_file_name)
    
    # Identify all the files to be delivered in drop
    drop_delivery_files = absoluteFilePaths(drop_delivery_dir)
    
    # Generate bom file
    out_file =  open(bom_file_name, "w")
    header = "DeliveryName\tInstallName\tFileCheckSum\tFileOrigin\tInstalledFilePermission\n"
    last_line = "#***Intel Confidential - Internal Use Only***\n"
    file_origin = "internal"
    out_file.write(header)
    for f in drop_delivery_files:
        # skip bomfile entry in bob file
        if f.endswith(bom_file_name):
            continue
        if component == 'tensorflow'  and "bom_tags" in f:
            continue
        delivery_name = f.replace(drop_delivery_dir,"<deliverydir>")
        if "conda_channel" in f:
            install_name = f.replace(drop_delivery_dir,"<installdir>")
        else:
            install_name = f.replace(drop_delivery_dir,"<installdir>/"+component)
        file_cksum = get_cksum(f)
        file_permission = "755" if f.endswith(".sh") or f.endswith(".py") else "644"
        line = delivery_name + "\t" + install_name + "\t" + file_cksum + "\t" + file_origin + "\t" + file_permission + "\n"
        out_file.write(line)
    out_file.write(last_line)
    
def main():
    print("-- AIKit Binary Drop --")
    parser = argparse.ArgumentParser()
    parser.add_argument('--drop_delivery_dir', help='component binary drop directory',
                        type=str, required=True)
    parser.add_argument('--boms_dir', help='directory where generated bom files will be stored',
                        type=str)
    parser.add_argument('--component', help='component name', type=str, default='modelzoo')
    args = parser.parse_args()

    drop_delivery_dir = args.drop_delivery_dir
    boms_dir = os.path.join(args.drop_delivery_dir, "boms")
    component = args.component
    bom_file_name = os.path.join(boms_dir, "l_"+component+".txt")
    
    print("drop_delivery_dir: ", drop_delivery_dir)
    print("boms_dir: ", boms_dir)
    print("component: ", component)
    print("bom_file: ", bom_file_name)
    generate_bom_file(bom_file_name, drop_delivery_dir, component)

if __name__ == "__main__":
    main()
