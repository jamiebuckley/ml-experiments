import os
from ftplib import FTP
from config import ROOT_DIR
import tarfile
import logging
from glob import glob
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AusAcornSatDataRetriever:
    csv_sat_path = os.path.join(ROOT_DIR, 'custom_datasets', 'aus_acorn_sites.csv')
    ftp_url = "ftp.bom.gov.au"
    ftp_urls = {
        'max': "/anon/home/ncc/www/change/ACORN_SAT_daily/acorn_sat_v2.6.0_daily_tmax.tar.gz",
        'min': "/anon/home/ncc/www/change/ACORN_SAT_daily/acorn_sat_v2.6.0_daily_tmin.tar.gz",
        'mean': "/anon/home/ncc/www/change/ACORN_SAT_daily/acorn_sat_v2.6.0_daily_tmean.tar.gz"
    }

    raw_data_path = os.path.join(ROOT_DIR, '.rawdata')
    raw_acorn_sat_path = os.path.join(raw_data_path, 'acorn_sat')

    def download(self, mode):
        self._ensure_raw_data_dir_exists()

        if mode not in self.ftp_urls.keys():
            raise ValueError(f"Invalid mode {mode}, expected one of {self.ftp_urls.keys()}")

        self._download_if_not_exists(mode)
        self._construct_dataset(mode)

    def _ensure_raw_data_dir_exists(self):
        if not os.path.exists(self.raw_acorn_sat_path):
            logger.info(f"Creating directory {self.raw_acorn_sat_path}")
            os.makedirs(self.raw_acorn_sat_path)
        else:
            logger.info(f"Directory {self.raw_acorn_sat_path} already exists")

    def _download_if_not_exists(self, mode):
        expected_file_path = os.path.join(self.raw_acorn_sat_path, mode + '.tar.gz')
        if not os.path.exists(expected_file_path):
            logger.info(f"Downloading {mode} from {self.ftp_url}")
            self._download_file_ftp(self.ftp_urls[mode], expected_file_path)
        else:
            logger.info(f"File {expected_file_path} already exists, skipping download")

        self._extract_file(expected_file_path, mode)


    def _download_file_ftp(self, url, file_path):
        ftp = FTP(self.ftp_url)
        ftp.login()

        parts = url.rsplit('/', 1)
        filename = parts[1]
        ftp.cwd(parts[0])
        with open(file_path, 'wb') as local_file:
            ftp.retrbinary('RETR ' + filename, local_file.write)
        ftp.quit()

    def _extract_file(self, file_path, mode):
        output_archive_dir = os.path.join(self.raw_acorn_sat_path, mode)

        if os.path.exists(output_archive_dir):
            logger.info(f"Directory {output_archive_dir} already exists, skipping extraction")
            return

        logger.info(f"Extracting {mode}")
        tar = tarfile.open(file_path)
        tar.extractall(path=output_archive_dir)
        logger.info(f"Extraction complete")

    def _construct_dataset(self, mode):
        parquet_path = os.path.join(self.raw_acorn_sat_path, mode + '.parquet')
        if os.path.exists(parquet_path):
            logger.info(f"File {parquet_path} already exists, skipping construction")
            return
        else:
            logger.info(f"Constructing {mode} dataset")

        sites = pd.read_csv(self.csv_sat_path, dtype={'stn_num': 'Int64'})
        folder_path = os.path.join(self.raw_acorn_sat_path, 'max')
        files = glob(os.path.join(folder_path, '*.csv'))

        datasets = []
        for file in files:
            column_names = ['date', mode, 'site', 'site_name']
            df = pd.read_csv(
                file,
                header=0,
                dtype={'site number': 'Int64'},
                names=column_names,
                parse_dates=['date']
            )

            site_number = df.iloc[0]['site']
            site_details = sites[sites['stn_num'] == site_number].iloc[0]

            # Weird format with the station details on the first row only
            df = df.iloc[1:].reset_index(drop=True)
            df.drop(['site', 'site_name'], axis=1, inplace=True)

            df['lat'] = site_details['lat']
            df['lon'] = site_details['lon']
            df['elevation'] = site_details['elevation']
            df['stn_num'] = site_details['stn_num']

            # filter rows with missing values
            df = df[df[mode].notna()]

            start_of_data_collection = pd.Timestamp('1910-01-01')
            df['days_since_start'] = (df['date'] - start_of_data_collection).dt.days
            df['days_since_start_of_year'] = df['date'].dt.dayofyear
            df['days_ssoy_sin'] = np.sin(2 * np.pi * df['days_since_start_of_year'] / 365)
            df['days_ssoy_cos'] = np.cos(2 * np.pi * df['days_since_start_of_year'] / 365)
            df.drop(['date'], axis=1, inplace=True)
            datasets.append(df)

        combined_df = pd.concat(datasets)
        combined_df.to_parquet(parquet_path, index=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    AusAcornSatDataRetriever().download(mode="max")
