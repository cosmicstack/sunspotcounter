#!/bin/bash
# COMMANDLINE ARGUMENTS
STARTDATE=$1
ENDDATE=$2
CHANNEL=$3
RESOLUTION=$4
DOWNLOAD_PATH=$5

# SDO WEBSITE URL
SDOURL=https://sdo.gsfc.nasa.gov
BROWSEDIR=$SDOURL"/assets/img/browse"

# DOWNLOAD PATH
LOCALDIR=$DOWNLOAD_PATH

# UNIX TIMESTAMPS
STARTSECONDS=$(date -j -u -f "%Y-%m-%d" ${STARTDATE} +"%s")
ENDSECONDS=$(date -j -u -f "%Y-%m-%d" ${ENDDATE} +"%s")

echo -e "\n\n"
echo "Download Images to local directory"
echo "START DATE: "$STARTDATE
echo "END DATE: "$ENDDATE
echo "CHANNEL: "$CHANNEL
echo "RESOLUTION: "$RESOLUTION
echo "DOWNLOAD PATH: "$LOCALDIR
echo -e "\n"

val=0
for (( i=$STARTSECONDS; i<=$ENDSECONDS; i+=86400 ))
do
	NEXTDATEPATH=$(date -j -u -f %s "${i}" +%Y/%m/%d)
	NEXTDATESTRING=$(date -j -u -f %s "${i}" +%Y%m%d)
	URL=${BROWSEDIR}/${NEXTDATEPATH}
	ACCEPT=${NEXTDATESTRING}_11*_${RESOLUTION}_${CHANNEL}.jpg
	printf "Downloading Images from: %s\r" "$URL"
	wget -q -nd --no-check-certificate --level=1 --recursive -e robots=off --no-parent -R "index.html*" -A $ACCEPT $URL --directory-prefix=$LOCALDIR
    
done

echo -e "\n"Script complete: $(date)