#!/bin/sh

# mkdir checkpoint
# cd checkpoint
# Download checkpoints
# -
# How to download GDrive files: https://stackoverflow.com/questions/37453841/download-a-file-from-google-drive-using-wget

if ! [ -f checkpoint/encoder.pt ]; then
    echo "Missing checkpoints. Downloading..."


    # wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt

    # Getting checkpoints zip stored in Crypsis drive
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OYwpYYise3JUdwFL9x1UuHodkdDsG1oz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OYwpYYise3JUdwFL9x1UuHodkdDsG1oz" -O checkpoint.zip && rm -rf /tmp/cookies.txt

    if ! [ -f checkpoint.zip ]; then
        echo "Download from first source failed. Downloading again from another URL..."

        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_7jRYtNfAn8EkNgYsN7_O8HFxPrEfl1j' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_7jRYtNfAn8EkNgYsN7_O8HFxPrEfl1j" -O checkpoint.zip && rm -rf /tmp/cookies.txt

    fi

unzip checkpoint.zip

rm checkpoint.zip

fi
