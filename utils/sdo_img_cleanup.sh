#!/bin/bash

DIRECTORY=$1  # Pass the directory containing images as an argument

# Create a list of unique dates using find
find "${DIRECTORY}" -name "*.jpg" -exec basename {} \; | cut -c1-8 | sort | uniq | while read DATE
do
    # Get first image matching this date using find
    FIRST_IMAGE=$(find "${DIRECTORY}" -name "${DATE}*.jpg" -print | head -n 1)
    
    # Extract components for renaming
    FILENAME=$(basename "$FIRST_IMAGE")
    NEW_NAME="${DATE}_$(echo $FILENAME | cut -d'_' -f3,4)"
    
    # Rename the file we're keeping
    mv "$FIRST_IMAGE" "${DIRECTORY}/${NEW_NAME}"
    
    # Remove all other images for this date using find
    find "${DIRECTORY}" -name "${DATE}*.jpg" -not -name "${NEW_NAME}" -delete
    
    # Progress indicator
    echo "Processed date: $DATE"
done

echo "Cleanup complete. Files renamed and only one image per day retained."