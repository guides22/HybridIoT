"""
Defines Position: class to represent a position in 2D space.
Longitude and latitude are used for geospatial coordinates, while X and Y are used for grid-based coordinates.
"""
class Position:

    def __init__(self, longitude: float=0.0, latitude: float=0.0, altitude: float=0.0, X: int=0, Y: int=0):
        """
        Initialize a Position with given longitude, latitude, altitude, X, and Y coordinates.
        """
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude
        self.X = X
        self.Y = Y


    def __str__(self):
        return f"Longitude: {self.longitude}, Latitude: {self.latitude}, Altitude: {self.altitude}, X: {self.X}, Y: {self.Y}"
    
    def __repr__(self):
        return f"Position(lon={self.longitude}, lat={self.latitude}, alt={self.altitude}, X={self.X}, Y={self.Y}))"