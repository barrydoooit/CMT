#!/bin/bash

# Function to create symbolic link
create_link() {
    local source_path=$1
    local target_path=$2

    # Remove the existing symbolic link or directory, if it exists
    if [ -e "$target_path" ] || [ -L "$target_path" ]; then
        rm -rf "$target_path"
    fi

    # Create a new symbolic link
    ln -s "$source_path" "$target_path"
    echo "Symbolic link created at $target_path"
}

# Function to list the contents of the target directory
list_contents() {
    local target_dir=$1
    echo "Status of $target_dir:"
    ls -la "$target_dir"
    echo ""
}

# Base paths
nuscenes_cc_root="/CMT/data/nuScenes-CC"
nuscenes_cl_root="/CMT/data/nuScenes-CL"
nuscenes_target_root="/CMT/data/nuscenes/samples"

# Cameras
declare -a cameras=("CAM_BACK" "CAM_BACK_LEFT" "CAM_BACK_RIGHT" "CAM_FRONT" "CAM_FRONT_LEFT" "CAM_FRONT_RIGHT")

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cam) 
            cam_type="$2"; 
            shift; 
            if [ "$2" != "--lidar" ]; then
                cam_severity="$2"; 
                shift; 
            fi
        ;;
        --lidar) 
            lidar_type="$2"; 
            shift; 
            if [ "$2" ]; then
                lidar_severity="$2"; 
                shift; 
            fi
        ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Process camera data if specified
if [ ! -z "$cam_type" ]; then
    for cam in "${cameras[@]}"; do
        if [ "$cam_type" = "clear" ]; then
            cam_path="${nuscenes_cc_root}/clear/samples/${cam}"
        elif [ "$cam_type" = "pure_noise" ]; then
            cam_path="${nuscenes_cc_root}/pure_noise/samples/${cam}"
        else
            cam_path="${nuscenes_cc_root}/${cam_type}/${cam_severity}/samples/${cam}"
        fi
        target_cam_path="${nuscenes_target_root}/${cam}"
        create_link "$cam_path" "$target_cam_path"
    done
fi

# Process LiDAR data if specified
if [ ! -z "$lidar_type" ]; then
    if [ "$lidar_type" = "clear" ]; then
        lidar_path="${nuscenes_cl_root}/clear/samples/LIDAR_TOP"
    else
        lidar_path="${nuscenes_cl_root}/${lidar_type}/${lidar_severity}/samples/LIDAR_TOP"
    fi
    target_lidar_path="${nuscenes_target_root}/LIDAR_TOP"
    create_link "$lidar_path" "$target_lidar_path"
fi

list_contents "$nuscenes_target_root"
