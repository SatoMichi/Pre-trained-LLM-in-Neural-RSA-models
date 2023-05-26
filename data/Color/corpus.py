from collections import Counter, defaultdict
import csv
import colorsys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

TURN_BOUNDARY =  " ### "

class ColorsCorpusReader:
    def __init__(self, src_filename, word_count=None, normalize_colors=True):
        self.src_filename = src_filename
        self.word_count = word_count
        self.normalize_colors = normalize_colors

    # () -> ColorsCorpusExample
    def read(self):
        grouped = defaultdict(list)
        with open(self.src_filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['role'] == 'speaker' and self._word_count_filter(row):
                    grouped[(row['gameid'], row['roundNum'])].append(row)
        for rows in grouped.values():
            yield ColorsCorpusExample(rows, normalize_colors=self.normalize_colors)

    def _word_count_filter(self, row):
        return self.word_count is None or row['contents'].count(" ") == (self.word_count-1)


class ColorsCorpusExample:
    def __init__(self, rows, normalize_colors=True):
        self.normalize_colors = normalize_colors
        self.contents = TURN_BOUNDARY.join([r['contents'] for r in rows])
        # Make sure our assumptions about these rows are correct:
        self._check_row_alignment(rows)
        row = rows[0]
        self.gameid = row['gameid']
        self.roundNum = int(row['roundNum'])
        self.condition = row['condition']
        self.outcome = row['outcome'] == 'true'
        self.clickStatus = row['clickStatus']
        self.color_data = []
        for typ in ['click', 'alt1', 'alt2']:
            self.color_data.append({
                'type': typ,
                'Status': row['{}Status'.format(typ)],
                'rep': self._get_color_rep(row, typ),
                'speaker': int(row['{}LocS'.format(typ)]),
                'listener': int(row['{}LocL'.format(typ)])})
        self.colors = self._get_reps_in_order('Status')
        self.listener_context = self._get_reps_in_order('listener')
        self.speaker_context = self._get_reps_in_order('speaker')

    def get_context_data(self):
        rgbs = [self._convert_hls_to_rgb(*c) for c in self.colors]
        context = self.parse_turns()[0]
        return (rgbs,context)

    def get_l0_data(self):
        rgbs = [self._convert_hls_to_rgb(*c) for c in self.colors]
        context = self.parse_turns()
        return (rgbs[2],context)

    def get_l0_negative_data(self):
        rgbs = [self._convert_hls_to_rgb(*c) for c in self.colors]
        context = self.parse_turns()
        return (rgbs[0],context)
    
    def get_l0_negative_data_another(self):
        rgbs = [self._convert_hls_to_rgb(*c) for c in self.colors]
        context = self.parse_turns()
        return (rgbs[1],context)
    
    # () -> [str]
    def parse_turns(self):
        return self.contents.split(TURN_BOUNDARY)

    def display(self, typ='model'):
        #print(self.contents)
        if typ == 'model':
            colors = self.colors
            target_index = 2
        elif typ == 'listener':
            colors = self.listener_context
            target_index = None
        elif typ == 'speaker':
            colors = self.speaker_context
            target_index = self._get_target_index('speaker')
        else:
            raise ValueError('`typ` options: "model", "listener", "speaker"')
        rgbs = [self._convert_hls_to_rgb(*c) for c in colors]
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 2))
        fig.suptitle(self.contents)
        for i, c in enumerate(rgbs):
            ec = c if (i != target_index or typ == 'listener') else "black"
            patch = mpatch.Rectangle((0, 0), 1, 1, color=c, ec=ec, lw=8)
            axes[i].add_patch(patch)
            axes[i].axis('off')
        plt.show()

    # (row,str) -> [float]
    def _get_color_rep(self, row, typ):
        rep = []
        for dim in ['H', 'L', 'S']:
            colname = "{}Col{}".format(typ, dim)
            rep.append(float(row[colname]))
        if self.normalize_colors: rep = self._scale_color(*rep)
        return rep

    def _convert_hls_to_rgb(self, h, l, s):
        if not self.normalize_colors: h, l, s = self._scale_color(h, l, s)
        return colorsys.hls_to_rgb(h, l, s)

    @staticmethod
    def _scale_color(h, l, s):
        return [h/360, l/100, s/100]

    # str -> [[float]]
    def _get_reps_in_order(self, field):
        colors = [(d[field], d['rep']) for d in self.color_data]
        return [rep for s, rep in sorted(colors)]

    def _get_target_index(self, field):
        for d in self.color_data:
            if d['Status'] == 'target': return d[field] - 1

    @staticmethod
    def _check_row_alignment(rows):
        keys = set(rows[0].keys())
        for row in rows[1:]:
            if set(row.keys()) != keys:
                raise RuntimeError("The dicts in the `rows` argument to `ColorsCorpusExample` must have all the same keys.")
        exempted = {'contents', 'msgTime','numRawWords', 'numRawChars','numCleanWords', 'numCleanChars'}
        keys = keys - exempted
        for row in rows[1: ]:
            for key in keys:
                if rows[0][key] != row[key]:
                    raise RuntimeError(
                        "The dicts in the `rows` argument to `ColorsCorpusExample` "
                        "must have all the same key values except for the keys "
                        "associated with the message. The key {} has values {} "
                        "and {}".format(key, rows[0][key], row[key]))

    def __str__(self):
        return self.contents


if __name__ == '__main__':
    from pathlib import Path
    import os

    root = Path(__file__).parent.parent.absolute()
    data_path = os.path.join(root,"data")
    print(data_path)
    corpus = ColorsCorpusReader(os.path.join(data_path,"colors.csv"), word_count=None, normalize_colors=True)
    examples = list(corpus.read())
    print("Number of datapoints: {}".format(len(examples)))
    ex1 = next(corpus.read())
    ex1.display()
    ex1.display(typ='listener')
    ex1.display(typ='speaker')