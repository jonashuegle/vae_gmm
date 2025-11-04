import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb, rgb_to_hsv, hsv_to_rgb
from mpl_toolkits.basemap import Basemap
import numpy as np
import xarray as xr
import os



def draw_custom_lon_labels(ax, m, boundinglat=30, offset=3, tick_degrees=None, fontsize=8, fontname='serif'):
    if tick_degrees is None:
        tick_degrees = np.arange(0, 361, 30)
    for lon in tick_degrees:
        lat = boundinglat + offset
        x, y = m(lon, lat)
        deg_label = int((lon + 360) % 360)
        angle = lon
        text_angle = angle
        if 90 < (angle % 360) < 270:
            text_angle += 180
        ax.text(
            x, y, f"{deg_label}°",
            fontsize=fontsize,
            fontname=fontname,
            rotation=text_angle,
            ha='center', va='center',
            rotation_mode='anchor',
            clip_on=False,
        )





class Plotting:
    def __init__(self, lon, lat, projection='npstere', lon_0=0,
                max_scale = 20,
                levels = np.array([-20, -15, -10, -8, -3, -2, -1, 1, 2, 3, 8, 10, 15, 20]),
                base_colors = [(0.0, "blue"), (0.5, "white"), (1.0, "red")],
                adjustment_factors = {"blue": [3,3], "white": [1,1], "red": [3,3],},
                adjustment_factors_border = {"blue": [100,100], "white": [1,1], "red": [100,100]},
                fontsize = 10,
                padding = 0.3,
                shrinking = 0.7,
            ):

        self.lon, self.lat = np.meshgrid(lon, lat)

        self.projection = projection
        self.cmap = self.create_custom_cmap(max_scale, base_colors, adjustment_factors)
        self.cmap_border = self.create_custom_cmap(max_scale, base_colors, adjustment_factors_border)
        self.levels = levels
        self.norm = Normalize(vmin=levels.min(), vmax=levels.max(), clip=False)

        self.lon_0 = lon_0
        self.fig, self.axs = None, None
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


    def plot_isolines(
        self, data, fig=None, axes=None, titles=None, show_colorbar=True, colorbar_kwargs=None
    ):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        n_clusters = data.shape[0]
        if axes is None:
            fig, axs = plt.subplots(nrows=1, ncols=n_clusters, figsize=(5*n_clusters,5))
            axs = [axs] if n_clusters == 1 else np.ravel(axs).tolist()
            _own_axes = True
        else:
            axs = axes if isinstance(axes, (list, np.ndarray)) else [axes]
            _own_axes = False

        if titles is None:
            titles = [f"Cluster {i+1}" for i in range(n_clusters)]
        elif isinstance(titles, str):
            titles = [titles] * n_clusters

        tick_size = plt.rcParams.get("xtick.labelsize", 8)
        font_family = plt.rcParams.get("font.family", "serif")
        font_name = font_family[0] if isinstance(font_family, (list, tuple)) else font_family
        cf_handles = []
        for i, ax in enumerate(axs):
            m = Basemap(projection=self.projection, boundinglat=30, lon_0=self.lon_0,
                        resolution='i', round=True, ax=ax)
            x, y = m(self.lon, self.lat)
            m.contour(x, y, data[i], levels=self.levels, colors='black', linewidths=0.3, ax=ax)
            cf = m.contourf(x, y, data[i], levels=self.levels, cmap=self.cmap, norm=self.norm, alpha=1, ax=ax)
            cf_handles.append(cf)
            m.drawcoastlines(linewidth=0.2)
            # Nur noch Parallelen automatisch labeln lassen, Meridiane OHNE Label:
            parallels = m.drawparallels(np.arange(30., 90., 30.), labels=[1, 0, 0, 0], linewidth=0.3)
            m.drawmeridians(np.arange(-180., 181., 30.), labels=[0, 0, 0, 0], linewidth=0.3)  # alle Labels aus
            # Parallelen: wie gehabt
            for lat, label_group in parallels.items():
                for label in label_group[1]:
                    label.set_fontsize(tick_size)
                    label.set_fontname(font_name)

            # ----- NEU: Eigene Longitude-Labels -----
            # Nach dem Zeichnen des Plots:
            draw_custom_lon_labels(ax, m, boundinglat=30, offset=-5, fontsize=8, fontname='serif', tick_degrees=[270, 300, 330, 0, 30, 60, 90])

            # -----------------------------------------

            ax.set_title(titles[i], pad=2, fontsize=8)
            ax.set_frame_on(False)
            ax.set_ylim(-100000, 7.4e6)

        # Colorbar nur, wenn alles intern erzeugt wurde
        if _own_axes and show_colorbar:
            colorbar_kwargs = colorbar_kwargs or {}
            cb = fig.colorbar(
                cf_handles[0], ax=axs, orientation='horizontal',
                fraction=0.04, pad=0.08, **colorbar_kwargs
            )
            cb.set_label('[hPa]')
            cb.set_ticks(self.levels)
            cb.set_ticklabels([str(lvl) for lvl in self.levels])
        else:
            cb = None

        #fig.tight_layout()

        return fig, axs, cf_handles, cb



    # def plot_isolines(
    #     self, data, fig=None, axes=None, titles=None, show_colorbar=True, colorbar_kwargs=None
    # ):
    #     if data.ndim == 2:
    #         data = data[np.newaxis, ...]
    #     n_clusters = data.shape[0]
    #     if fig is None or axes is None:
    #         fig, axs = plt.subplots(nrows=1, ncols=n_clusters, figsize=(5*n_clusters,5))
    #         axs = [axs] if n_clusters == 1 else np.ravel(axs).tolist()
    #         _own_axes = True
    #     else:
    #         axs = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    #         _own_axes = False

    #     if titles is None:
    #         titles = [f"Cluster {i+1}" for i in range(n_clusters)]
    #     elif isinstance(titles, str):
    #         titles = [titles] * n_clusters

    #     tick_size = plt.rcParams.get("xtick.labelsize", 8)
    #     font_family = plt.rcParams.get("font.family", "serif")
    #     font_name = font_family[0] if isinstance(font_family, (list, tuple)) else font_family
    #     cf_handles = []
    #     for i, ax in enumerate(axs):
    #         m = Basemap(projection=self.projection, boundinglat=30, lon_0=self.lon_0,
    #                     resolution='i', round=True, ax=ax)
    #         x, y = m(self.lon, self.lat)
    #         m.contour(x, y, data[i], levels=self.levels, colors='black', linewidths=0.8, ax=ax)
    #         cf = m.contourf(x, y, data[i], levels=self.levels, cmap=self.cmap, norm=self.norm, alpha=1, ax=ax)
    #         cf_handles.append(cf)
    #         m.drawcoastlines(linewidth=0.2)
    #         parallels = m.drawparallels(np.arange(30., 90., 30.), labels=[1, 0, 0, 0])
    #         meridians = m.drawmeridians(np.arange(-180., 181., 30.), labels=[0, 0, 0, 1])
            
    #         # Parallelen: nur Größe/Font anpassen
    #         for lat, label_group in parallels.items():
    #             for label in label_group[1]:
    #                 label.set_fontsize(tick_size)
    #                 label.set_fontname(font_name)
            
    #         for lon, label_group in meridians.items():
    #             for label in label_group[1]:
    #                 degree = int(lon)
    #                 if degree < 0:
    #                     degree = 360 + degree
    #                 label.set_text(f"{degree}°")
    #                 label.set_fontsize(tick_size)
    #                 label.set_fontname(font_name)
    #                 # Probier beide aus, welche optisch besser passt:
    #                 #label.set_rotation((degree + 180) % 360)
    #                 label.set_rotation(degree)
    #                 x, y = label.get_position()
    #                 r_offset = 0.1  # Passe an!
    #                 label.set_position((x, y + r_offset))

    #                 label.set_va('center')
    #                 label.set_ha('center')
                    



    #         ax.set_title(titles[i], pad=2)
    #         ax.set_frame_on(False)
    #         ax.set_ylim(0, 7.4e6)

    #     # Colorbar nur, wenn alles intern erzeugt wurde
    #     if _own_axes and show_colorbar:
    #         colorbar_kwargs = colorbar_kwargs or {}
    #         cb = fig.colorbar(
    #             cf_handles[0], ax=axs, orientation='horizontal',
    #             fraction=0.04, pad=0.08, **colorbar_kwargs
    #         )
    #         cb.set_label('[hPa]')
    #         cb.set_ticks(self.levels)
    #         cb.set_ticklabels([str(lvl) for lvl in self.levels])
    #     else:
    #         cb = None

    #     return fig, axs, cf_handles, cb





    def plot_data(self, data, ncols=None):
        if ncols is None:
            ncols = data.shape[0]
        nrows = int(np.ceil(data.shape[0] / ncols))
        self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.fig_sizes)
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
         