import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb, rgb_to_hsv, hsv_to_rgb
from mpl_toolkits.basemap import Basemap
import numpy as np
import xarray as xr
import os
class Plotting:
    def __init__(self, lon, lat, projection='npstere', lon_0=0, fig_scale_factor = 4,
                max_scale = 20,
                levels = np.array([-20 ,-15, -10, -8, -3, -2, -1, 1, 2, 3, 8, 10, 15, 20]),
                base_colors = [(0.0, "blue"), (0.5, "white"), (1.0, "red")],
                adjustment_factors = {"blue": [3,3], "white": [1,1], "red": [3,3],},
                adjustment_factors_border = {"blue": [100,100], "white": [1,1], "red": [100,100]},
                fontsize = 20,
                padding = 0.3,
                shrinking = 0.7,
            ):

        self.lon, self.lat = np.meshgrid(lon, lat)

        self.projection = projection
        self.cmap = self.create_custom_cmap(max_scale, base_colors, adjustment_factors)
        self.cmap_border = self.create_custom_cmap(max_scale, base_colors, adjustment_factors_border)
        self.levels = levels
        self.norm = Normalize(vmin=-max_scale, vmax=max_scale, clip=False)

        self.lon_0 = lon_0
        self.fig, self.axs = None, None
        self.fig_scale_factor = fig_scale_factor
        plt.rcParams.update({'font.size':fontsize})
        self.padding = padding
        self.shrinking = shrinking

    def _get_label(self, meridian):
        if meridian < 0:
            return str(int(360 + meridian))+'°'
        if meridian == 0:
            return '0°'
        if meridian > 0:
            return str(int(meridian))+'°'
        

    def create_custom_cmap(self, max_scale, base_colors, adjustment_factors=None):
        """
        Erstellt eine angepasste Colormap mit einstellbarer Sättigung für jede Farbe.

        :param max_scale: Skalierungsfaktor für die Positionsberechnung der Farben.
        :param base_colors: Eine Liste von Tupeln, die die relative Position und Farben darstellen.
                            Beispiel: [(0.0, "blue"), (0.5, "white"), (1.0, "red")]
        :param saturation_factors: Ein Dictionary, das angibt, wie die Sättigung jeder Farbe angepasst werden soll.
                                Werte kleiner als 1 reduzieren die Sättigung, Werte größer als 1 erhöhen sie.
                                Beispiel: {"blue": 0.5, "white": 0.2, "red": 1.5}
        :return: Ein Colormap-Objekt
        """
        if adjustment_factors is None:
            adjustment_factors = {}

        def adjust_color_properties(color, factors):
            # Konvertiere RGB zu HSV, passe die Sättigung und Helligkeit an, konvertiere zurück zu RGB
            rgb = to_rgb(color)
            hsv = rgb_to_hsv(rgb)
            saturation_factor, brightness_factor = factors.get(color, [1, 1])
            hsv[1] = max(0, min(1, hsv[1] * saturation_factor))  # Sättigung sicher im Bereich [0, 1] halten
            hsv[2] = max(0, min(1, hsv[2] * brightness_factor))  # Helligkeit sicher im Bereich [0, 1] halten
            return hsv_to_rgb(hsv)

        # Berechne die neuen Farbpositionen basierend auf max_scale
        adjusted_colors = [(pos, adjust_color_properties(color, adjustment_factors)) 
                        for pos, color in base_colors]
        cmap = LinearSegmentedColormap.from_list("CustomCmap", adjusted_colors)
        return cmap

        
    
    def set_extent(self, extent_name):
        if extent_name in self.extents:
            self.current_extent = extent_name
        else:
            print(f"Warning: {extent_name} not found in extents dictionary. Using default.")
            self.current_extent = "default"



    def plot_isolines(self, data, ncols=None, individual_colorbar=False, save_dir=None, dpi=300, titles=None):
        """
        Erstellt Isolinienplots und speichert sie optional als PNG.

        :param data: Array mit Isolinienwerten. Erwartet entweder
                     - 3D-Array [n_clusters, lat, lon] oder
                     - 2D-Array [lat, lon] (wird als einzelner Cluster interpretiert).
        :param ncols: Anzahl der Spalten in der Subplot-Anordnung.
        :param individual_colorbar: Falls True, erhält jeder Plot (bzw. die jeweilige Figure) eine eigene Colorbar.
        :param save_dir: Falls angegeben, werden die einzelnen Plots in diesem Verzeichnis gespeichert.
        :param dpi: Auflösung der gespeicherten Bilder.
        :param titles: Optional. Entweder ein einzelner String, der für alle verwendet wird,
                       oder eine Liste von Strings (eine pro Cluster). Falls None, wird der Standardtitel "Cluster {i+1}" verwendet.
        :return: Tuple (fig, axs) – falls kein save_dir gesetzt ist, sonst (None, None).
        """

        # Falls data ein 2D-Array ist, erweitern zu einem einzelnen Cluster.
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # shape wird zu (1, lat, lon)

        n_clusters = data.shape[0]

        # Falls ncols nicht gesetzt ist und kein save_dir existiert, erstelle eine gemeinsame Figure mit Subplots.
        if save_dir is None:
            if ncols is None:
                ncols = n_clusters
            nrows = int(np.ceil(n_clusters / ncols))
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                                    figsize=(self.fig_scale_factor * ncols, self.fig_scale_factor * nrows))
            axs = np.atleast_1d(axs).flatten()
        else:
            fig, axs = None, None
            #os.makedirs(save_dir, exist_ok=True)

        for i in range(n_clusters):
            if save_dir:
                fig, ax = plt.subplots(figsize=(6, 6))
            else:
                ax = axs[i]
                fig = ax.figure

            m = Basemap(projection=self.projection, boundinglat=30, lon_0=self.lon_0,
                        resolution='i', round=True, ax=ax)
            x, y = m(self.lon, self.lat)

            cs = m.contour(x, y, data[i], levels=self.levels, colors='black', linewidths=0.8, ax=ax)
            cf = m.contourf(x, y, data[i], levels=self.levels, cmap=self.cmap, norm=self.norm, alpha=1, ax=ax)

            m.drawcoastlines(linewidth=0.2)
            m.drawparallels(np.arange(30., 90., 30.), labels=[1, 0, 0, 0])
            m.drawmeridians(np.arange(-180., 181., 30.), labels=[0, 0, 0, 1])

            # Setze den Titel: Falls titles nicht angegeben, verwende den Default, ansonsten
            # überprüfe, ob titles eine Liste/tuple ist oder ein einzelner String.
            if titles is None:
                title_str = f"Cluster {i+1}"
            else:
                if isinstance(titles, (list, tuple)):
                    title_str = titles[i] if i < len(titles) else f"Cluster {i+1}"
                else:
                    title_str = titles
            if title_str:  # Falls title_str nicht leer ist, setze den Titel.
                ax.set_title(title_str)

            ax.set_frame_on(False)
            ax.set_ylim(0, 7.4e6)

            if individual_colorbar:
                cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', extend='both', pad=0.1, shrink=1.2)
                cbar.set_label('[hPa]')
                cbar.set_ticks(self.levels)
                cbar.set_ticklabels([str(lvl) for lvl in self.levels])

            if save_dir:
                fig.savefig(save_dir+".png", dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"Gespeichert: {save_dir}.png")

        # Falls kein save_dir, keine globale Colorbar hinzufügen (da jedes Plot individuell gesteuert ist)
        return (fig, axs) if not save_dir else (None, None)





    def plot_data(self, data, ncols=None):
        if ncols is None:
            ncols = data.shape[0]
        nrows = int(np.ceil(data.shape[0] / ncols))
        self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(self.fig_scale_factor*ncols, self.fig_scale_factor*nrows))
        self.axs = self.axs.flatten()

        for i, ax in enumerate(self.axs):
            m = Basemap(projection=self.projection, boundinglat=30, lon_0=self.lon_0,
                        resolution='i', round=True, ax=ax)
            x, y = m(self.lon, self.lat)

            m.pcolormesh(x, y, data[i], cmap=self.cmap, norm=self.norm, ax=ax)

            m.drawcoastlines(linewidth=0.4)
            m.drawparallels(np.arange(30., 90., 30.), labels=[1, 0, 0, 0])
            meridians = m.drawmeridians(np.arange(-180., 181., 30.), labels=[0, 0, 0, 1])

            for meridian, label_group in meridians.items():
                for label in label_group[1]:
                    if meridian % 90 == 0:
                        label.set_text(self._get_label(meridian))
                        label.set_rotation(meridian)
                    else:
                        label.set_text('')
            ax.set_frame_on(False)  # Rahmen ausblenden
            ax.set_ylim(0, 7.4e6) 
        try:
            cbar = self.fig.colorbar(ax=self.axs.ravel().tolist(), orientation='horizontal', extend='both', pad=0.1, shrink=0.7)
            if self.levels is not None:
                cbar.set_ticks(self.levels)
                cbar.set_ticklabels([str(lvl) for lvl in self.levels])
            cbar.set_label('[hPa]')
        except:
            pass
        
        return self.fig, self.axs
         