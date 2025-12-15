import kete
import psycopg2

import numpy as np
import astropy.units as u

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.wcs import WCS

from pipeline.parameters import Parameters

default_radius_threshold = 30 * u.arcsec
chip_fov_radius = 1 * u.deg

db_params = {
    "dbname": "asteroid",
    "user": "vedshah",  # Change to your PostgreSQL username
    "host": "localhost",  # Change if PostgreSQL is hosted remotely
    "port": "5432"  # Default PostgreSQL port
}

def download_and_update_db(time_jd, db_params=db_params):

    # Establish connection
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    # Load orbit data from the MPC
    mpc_obs = kete.mpc.fetch_known_orbit_data(force_download=True)

    # Convert that data to State objects.
    mpc_states = kete.mpc.table_to_states(mpc_obs)

    # Update the states based on the current time. 
    mpc_states_jd = kete.propagate_n_body(mpc_states, time_jd)

    # Update the states for the next day. This will be used to find the proper motion.
    mpc_states_jd_plus_1 = kete.propagate_n_body(mpc_states_jd, time_jd + 1)

    # Convert the states from ecliptic to equatorial to easily access the RA and Dec.
    for s_jd, s_jd_plus_1 in zip(mpc_states_jd, mpc_states_jd_plus_1):

        # Convert the state to equatorial coordinates.
        ra_jd = s_jd.as_equatorial.pos.ra 
        dec_jd = s_jd.as_equatorial.pos.dec 

        ra_jd_plus_1 = s_jd_plus_1.as_equatorial.pos.ra 
        dec_jd_plus_1 = s_jd_plus_1.as_equatorial.pos.dec 

        # Compute the average proper motion for the next day. In deg/day.
        pm_ra = ra_jd_plus_1 - ra_jd
        pm_dec = dec_jd_plus_1 - dec_jd
        
        # Add the state to the database. 
        # TODO: Add them to a temporary table first and then replace the main table.
        cursor.execute("INSERT INTO asteroids (designation, jd, ra, dec, pm_ra, pm_dec, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z) VALUES \
                       (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", \
                       (s_jd.desig, s_jd.jd, ra_jd, dec_jd, pm_ra, pm_dec, s_jd.pos.x, s_jd.pos.y, s_jd.pos.z, s_jd.vel.x, s_jd.vel.y, s_jd.vel.z))

    conn.commit()

    # Close the connection
    cursor.close()
    conn.close()

class ParsAsteroidChecker(Parameters):
    def __init__(self, **kwargs):
        super().__init__()

        self.radius_threshold = self.add_par(
            'radius_threshold',
            30,
            float,
            'Radius threshold in arcseconds for matching asteroids to sources.',
            critical=False
        )

        self._enforce_no_new_attrs = True
        self.override(kwargs)

    def get_process_name(self):
        return 'asteroid_checking'

class AsteroidChecker:

    def __init__(self, **kwargs):
        
        # Load the database table and convert to states
        self.pars = ParsAsteroidChecker(**kwargs) 
        self._load_states_from_db()

    def _load_states_from_db(self, db_params=db_params):

        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()

            # Only select objects such that ra/dec + proper motion is within the field of view.
            cursor.execute("SELECT * FROM asteroids;")
            rows = cursor.fetchall()

            # Convert rows into sate objects
            self.mpc_states = []
            for row in rows:
                
                designation, jd, ra, dec, pm_ra, pm_dec, position_x, position_y, position_z, velocity_x, velocity_y, velocity_z = row
                state = kete.State(designation, jd, kete.vector.Vector([position_x, position_y, position_z]), \
                                kete.vector.Vector([velocity_x, velocity_y, velocity_z]))
                self.mpc_states.append(state)
            
            # Close the connection
            cursor.close()
            conn.close()

        except Exception as e:
            print("Error connecting to database:", e)    

    def run(self, ds):
        
        # NOTE: This is the pseudocode structure of the dataset 'ds'.
        wcs, time = ds['wcs'], ds['time_jd']
        mpc_skycoords, mpc_designations = self._get_asteroid_list(wcs, time)

        sources = ds['sources']
        sources_skycoords = SkyCoord(sources['ra'], sources['dec'], unit="deg")

        matched_asteroids = self._cross_match_sources_with_asteroids(mpc_skycoords, mpc_designations, sources_skycoords)

    
    def _get_asteroid_list(self, frame_wcs, time_jd):

        def _kete_state_ra_dec(state, sun2earth):
            """
            Private method to wrap computing the coordinates of a single state
            """

            # change the coordinate frame from heliocentric to geocentric
            # then compute the geocentric coordinate
            obj_earth_pos = state.pos - sun2earth 
            state_vec_equitorial = obj_earth_pos.change_frame(kete.vector.Frames.Equatorial)
            return state_vec_equitorial.ra, state_vec_equitorial.dec


        sun2earth = kete.spice.get_state("Earth", time_jd)

        fov = kete.fov.RectangleFOV.from_wcs(frame_wcs, sun2earth)
        fov_center_ra, fov_center_dec = fov.pointing.as_equatorial.ra, fov.pointing.as_equatorial.dec
        target_skycoord =  SkyCoord(fov_center_ra, fov_center_dec, unit="deg")

        print(fov_center_ra, fov_center_dec)

        # Update the states based on the time using the two body approximation
        new_mpc_states_2body = kete.propagate_two_body(self.mpc_states, time_jd)
        ras, decs = np.array([_kete_state_ra_dec(state, sun2earth.pos) for state in new_mpc_states_2body]).T

        # Only get MPC matches that are withing 2*survey fov radius 
        mpc_skycoords_approx = SkyCoord(ras, decs, unit="deg")
        idxs = np.where(target_skycoord.separation(mpc_skycoords_approx) < chip_fov_radius)[0]

        # Now use the full n-body solution for all of these objects
        objs_win_first_cut = np.array(self.mpc_states)[idxs]
        print(f"Found {len(objs_win_first_cut)} w/in {2 * chip_fov_radius}, running full n-body on this subset...")

        states = kete.propagate_n_body(objs_win_first_cut, time_jd)

        if len(states) > 0:
            ras, decs = np.array([_kete_state_ra_dec(state, sun2earth.pos) for state in states]).T
            mpc_skycoords = SkyCoord(ras, decs, unit="deg")
            designations_list = [s.desig for s in objs_win_first_cut]

            return mpc_skycoords, designations_list
        else:

            return [], []

    def _cross_match_sources_with_asteroids(self, asteroid_sky_coords, asteroid_designations, sources_sky_coords, radius_threshold=default_radius_threshold):

        matched_asteroids = []
        idx, d2d, _ = sources_sky_coords.match_to_catalog_sky(asteroid_sky_coords)

        # Process and display the cross-match results
        for source, match_index, separation in zip(sources_sky_coords, idx, d2d):

            if separation < radius_threshold:
                matched_asteroids.append(asteroid_designations[match_index])
            else:
                matched_asteroids.append(None)
        
        return matched_asteroids