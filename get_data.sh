/usr/bin/env bash


top_url=https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/10m

[ -d data ]  || mkdir data
pushd data

for tile in 18_38 18_39 19_38 19_39 28_38 20_39; do
    tile_file=${tile}_10m_v4.1_dem.tif
    [ -f $tile_file ] && continue
    echo $tile_file
    wget ${top_url}/${tile}/${tile_file}
done


rm ArcticDEM.vrt
gdalbuildvrt ArcticDEM.vrt *10m*dem.tif
