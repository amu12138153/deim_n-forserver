"""Code-faithful, editable SVG/PDF and PNG schematics of RWA, GSQWCA and CQWG.

Run from the project: python figures/water_modules/draw_water_module_schematics.py
Only figure files are produced; model implementation is not modified.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'engine/mynewblock/water_token_fusion.py'
FONT_FILE = Path('C:/Windows/Fonts/msyh.ttc')
FONT = FontProperties(fname=str(FONT_FILE)) if FONT_FILE.exists() else FontProperties(family='DejaVu Sans')
plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'savefig.facecolor':'white'})

INK='#233344'
MUTED='#657487'
LINE='#526575'
COLORS={
    'water':('#eaf3fc','#3c77ac'),
    'visual':('#eaf5ef','#438667'),
    'relation':('#f1ecf8','#8162a6'),
    'router':('#fff2df','#c48d3d'),
    'operation':('#f2f5f8','#8293a4'),
    'sparse':('#fceceb','#ba716b'),
}


class Diagram:
    def __init__(self,name,title,subtitle,height,accent):
        self.name,self.width,self.height=name,1200,height
        self.fig,self.ax=plt.subplots(figsize=(10,height/120))
        self.fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        self.ax.set(xlim=(0,1200),ylim=(height,0),aspect='equal')
        self.ax.axis('off')
        self.nodes={}
        self.fit_labels=[]
        self.ax.plot([50,1150],[81,81],color=accent,lw=2.2)
        heading=self.text(50,33,title,size=23,weight='bold',align='left')
        self.fit_labels.append((heading,1100,45))
        self.text(50,63,subtitle,size=12,color=MUTED,align='left')

    def text(self,x,y,text,size=14,color=INK,weight='normal',align='center'):
        return self.ax.text(x,y,text,fontproperties=FONT,fontsize=size,color=color,
                     fontweight=weight,ha=align,va='center',linespacing=1.55,zorder=5)

    def box(self,id,x,y,w,h,label,kind='operation',size=14):
        fill,stroke=COLORS[kind]
        self.nodes[id]=(x,y,w,h)
        self.ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',
                                        linewidth=1.5,facecolor=fill,edgecolor=stroke,zorder=3))
        label_artist=self.text(x+w/2,y+h/2,label,size=size)
        self.fit_labels.append((label_artist,w-32,h-22))

    def circle(self,id,cx,cy,label,r=27):
        self.nodes[id]=(cx-r,cy-r,2*r,2*r)
        self.ax.add_patch(Circle((cx,cy),r,facecolor='white',edgecolor=LINE,lw=1.6,zorder=3))
        self.text(cx,cy,label,size=23)

    def anchor(self,id,side='bottom',fraction=.5):
        x,y,w,h=self.nodes[id]
        if side=='top': return (x+w*fraction,y)
        if side=='bottom': return (x+w*fraction,y+h)
        if side=='left': return (x,y+h*fraction)
        return (x+w,y+h*fraction)

    def edge(self,source,target,start='bottom',end='top',via=(),dashed=False,color=LINE,
             start_fraction=.5,end_fraction=.5,label=None,label_at=None):
        points=[self.anchor(source,start,start_fraction),*via,self.anchor(target,end,end_fraction)]
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        self.ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=13,
                                         lw=1.6,color=color,linestyle=(0,(4,3)) if dashed else '-',
                                         joinstyle='round',capstyle='round',zorder=2))
        if label:
            self.text(*label_at,label,size=11,color=color)

    def bridge(self,x,y,vertical=True):
        # A small wire jump explicitly denotes a crossing without a junction.
        if vertical:
            self.ax.plot([x,x],[y-12,y+12],color='white',lw=5,zorder=2.1)
            verts=[(x,y-12),(x+15,y-6),(x+15,y+6),(x,y+12)]
        else:
            self.ax.plot([x-12,x+12],[y,y],color='white',lw=5,zorder=2.1)
            verts=[(x-12,y),(x-6,y-15),(x+6,y-15),(x+12,y)]
        self.ax.add_patch(PathPatch(MplPath(verts,[MplPath.MOVETO]+[MplPath.CURVE4]*3),
                                   facecolor='none',edgecolor=LINE,lw=1.6,zorder=2.2))

    def footer(self,lines):
        self.ax.plot([50,1150],[self.height-135]*2,color='#dce3ea',lw=.8)
        for i,line in enumerate(lines):
            self.text(50,self.height-104+i*28,line,size=10.2,color=MUTED,align='left')

    def save(self):
        # Measure rendered labels to keep all text within the rounded rectangles.
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        scale_x=self.ax.bbox.width/self.width
        scale_y=self.ax.bbox.height/self.height
        for artist,width,height in self.fit_labels:
            bounds=artist.get_window_extent(renderer)
            factor=min(1,width*scale_x/bounds.width,height*scale_y/bounds.height)
            if factor<1: artist.set_fontsize(artist.get_fontsize()*factor*.98)
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        for artist,width,height in self.fit_labels:
            bounds=artist.get_window_extent(renderer)
            assert bounds.width<=width*scale_x+1,artist.get_text()
            assert bounds.height<=height*scale_y+1,artist.get_text()
        for extension in ('svg','pdf','png'):
            path=HERE/f'{self.name}.{extension}'
            self.fig.savefig(path,dpi=300)
        plt.close(self.fig)


def draw_rwa():
    d=Diagram('01_RWA','(a) Rank-aware Water Attention — RWA',
              '水质关系建模  |  Encoded water tokens in; enhanced water tokens out',1440,COLORS['relation'][1])
    d.box('W',450,110,300,70,'Water tokens W\n[B, 3, Dw]','water',15)
    d.box('qkv',450,250,300,90,'Linear Q / K / V\nSplit heads: [B, Hw, 3, dhw]','water',13)
    d.box('Y',450,420,300,120,'Positive linear attention\nQ+, K+ = ELU(Q,K) + 1\nY = Q+ (K+^T V / 3) / z','water',13)
    d.box('stats',820,420,320,120,'Center tokens → covariance\nRedundancy + concentration\nrank_stats: [B, Hw, 2]','relation',12)
    d.box('router',820,600,320,110,'Rank-aware router\nLinear → GELU → Linear\n2 → 16 → 1','router',13)
    d.box('rs',860,750,240,80,'sigmoid → relation_strength\n[B, Hw, 1, 1]','router',11)
    d.box('R',70,250,300,100,'Token relation MLP\nLinear → SiLU → Linear\nSplit: [B, Hw, 3, dhw]','relation',13)
    d.circle('weighted',220,750,'×')
    d.circle('fused',600,900,'+')
    d.box('proj',450,970,300,90,'Merge heads → Linear\nΔ = OutProj(F)\n[B, 3, Dw]','operation',13)
    d.box('scale',450,1120,300,80,'α × Δ\nLearnable α (init 0.01)','router',12)
    d.circle('residual',600,1260,'+')
    d.box('out',820,1225,320,70,'Enhanced water W_out\n[B, 3, Dw]','water',15)
    d.edge('W','qkv')
    d.edge('W','R',start='left',via=[(220,145)],end='top')
    d.edge('qkv','Y')
    d.edge('Y','stats',start='right',end='left')
    d.edge('stats','router')
    d.edge('router','rs')
    d.edge('R','weighted',via=[(220,360)],label='R = MLP(W)',label_at=(155,510))
    d.edge('rs','weighted',start='left',end='right',via=[(780,790),(780,690),(285,690),(285,750)],dashed=True,color=COLORS['router'][1])
    d.edge('weighted','fused',start='bottom',end='left',via=[(220,900)])
    d.edge('Y','fused')
    d.edge('fused','proj')
    d.edge('proj','scale')
    d.edge('scale','residual')
    d.edge('W','residual',start='left',end='left',via=[(30,145),(30,1260)],label='Identity W',label_at=(110,1090))
    d.edge('residual','out',start='right',end='left')
    d.bridge(600,690)
    d.text(600,575,'z = clamp(Q+ · mean(K+), ε)',size=10.5,color=MUTED)
    d.text(1050,570,'detach()',size=11,color=MUTED)
    d.footer(['Dw = 64, Hw = 4, dhw = 16; relation_strength controls the relation MLP; α scales the final residual.',
              'Rank statistics are detached before the router. Spectral effective rank is diagnostic only.',
              'Solid arrows: feature flow. Dashed arrows: learned controls. A wire jump is not a connection.'])
    d.save()


def draw_gsqwca():
    d=Diagram('02_GSQWCA','(b) Global-Sparse Query-Water Cross Attention — GSQWCA',
              '全局–稀疏交叉注意力  |  Dense context + sparse context + query-adaptive branch routing',1740,COLORS['water'][1])
    d.box('Qin',130,110,280,70,'Object Query Q_in\n[B, N, Dq]','visual',15)
    d.box('W',790,110,280,70,'Enhanced water W\n[B, 3, Dw]','water',14)
    d.box('Q',130,250,280,90,'q_proj → split heads\nQ: [B, H, N, dh]','visual',14)
    d.box('wp',790,250,280,80,'water_proj: Linear(Dw,Dq)\n[B, 3, Dq]','water',13)
    d.box('KV',790,400,280,85,'k_proj / v_proj → split\nK,V: [B, H, 3, dh]','water',13)
    d.box('scores',450,510,300,110,'Scaled similarity scores\nS = Q K^T / sqrt(dh)\n[B, H, N, 3]','operation',13)
    d.box('Ag',100,690,330,100,'Global dense branch\nA_g = softmax(S)\nAll 3 water tokens retained','water',14)
    d.box('As',770,690,330,100,'Sparse branch\nTop-K mask → −∞ → softmax\nDefault K = 2; ablation K = 1','sparse',13)
    d.box('Fg',100,870,330,90,'A_g × V → merge heads\nF_g: [B, N, Dq]','water',14)
    d.box('Fs',770,870,330,90,'A_s × V → merge heads\nF_s: [B, N, Dq]','sparse',14)
    d.box('router',450,1040,300,140,'Concat(Q_in, F_g, F_s)\n[B, N, 3Dq]\nLinear(3Dq,Dq) → SiLU\n→ Linear(Dq,2) → softmax','router',12.5)
    d.box('weights',450,1230,300,70,'w_g, w_s: [B, N, 1]\nw_g + w_s = 1','router',14)
    d.box('mix',450,1360,300,75,'F = w_g ⊙ F_g + w_s ⊙ F_s\n[B, N, Dq]','operation',13)
    d.box('out',450,1480,300,75,'out_proj → Dropout\nWater context [B, N, Dq]','water',14)
    d.edge('Qin','Q')
    d.edge('W','wp')
    d.edge('wp','KV')
    d.edge('Q','scores',end='left',via=[(270,565)])
    d.edge('KV','scores',end='right',via=[(930,565)],label='K',label_at=(820,538))
    d.edge('scores','Ag',via=[(600,645),(265,645)])
    d.edge('scores','As',via=[(600,645),(935,645)])
    d.edge('Ag','Fg')
    d.edge('As','Fs')
    d.edge('Fg','router',start='right',end='left',via=[(440,915),(440,1080)],end_fraction=.28)
    d.edge('Fs','router',start='left',end='right',via=[(760,915),(760,1080)],end_fraction=.28)
    d.edge('Qin','router',start='left',end='left',via=[(40,145),(40,1140)],end_fraction=.71,color=COLORS['visual'][1],label='Q_in',label_at=(104,1115))
    d.edge('router','weights')
    d.edge('weights','mix')
    d.edge('Fg','mix',end='left',via=[(265,1398)])
    d.edge('Fs','mix',end='right',via=[(935,1398)])
    d.edge('mix','out')
    d.bridge(265,1140)
    d.footer(['Dq = 128, Dw = 64, H = 8, dh = 16; each decoder layer has its own query-adaptive router.',
              'Router final Linear weight/bias = 0 at initialization: w_g = w_s = 0.5 for every query.',
              'This module returns context only; query injection is performed by CQWG. Shared K,V serve both branches.'])
    d.save()


def draw_cqwg():
    d=Diagram('03_CQWG','(c) Complementary Query-Water Gate — CQWG',
              'Query 与通道级互补门控  |  Residual update of the original object query',1450,COLORS['visual'][1])
    d.box('Q',180,110,280,70,'Original Query Q\n[B, N, Dq]','visual',15)
    d.box('Fw',780,110,280,70,'Water context Fw\n[B, N, Dq]','water',15)
    d.box('features',450,300,400,110,'Concat(Q, Fw, Q ⊙ Fw, |Q − Fw|)\nGate input [B, N, 4Dq]','operation',14)
    d.box('diff',100,300,270,110,'Complementary difference\nFw − Q\n[B, N, Dq]','operation',12)
    d.box('mlp',450,480,400,100,'Linear(4Dq,Dq) → SiLU\n→ Linear(Dq,Dq)','router',15)
    d.box('gate',450,650,400,80,'Sigmoid → G [B, N, Dq]\nQuery-specific and channel-specific','router',13)
    d.box('delta',450,800,400,80,'Q_mix − Q = G ⊙ (Fw − Q)\n[B, N, Dq]','operation',15)
    d.box('norm',450,960,400,75,'LayerNorm on the difference\nΔ = LN(G ⊙ (Fw − Q))','operation',15)
    d.box('scale',450,1110,400,75,'λ_gate × Δ\nLearnable λ_gate: init 0.01','router',15)
    d.circle('plus',650,1240,'+')
    d.box('out',890,1202,270,75,'Updated Query Q_out\n[B, N, Dq]','visual',15)
    d.edge('Q','features',via=[(320,245),(650,245)])
    d.edge('Fw','features',end='right',via=[(920,355)])
    d.edge('Q','diff',start='left',end='left',via=[(75,145),(75,355)])
    d.edge('Fw','diff',start='left',via=[(740,145),(740,215),(235,215)])
    d.edge('features','mlp')
    d.edge('mlp','gate')
    d.edge('gate','delta')
    d.edge('diff','delta',end='left',via=[(235,840)])
    d.edge('delta','norm')
    d.edge('norm','scale')
    d.edge('scale','plus')
    d.edge('Q','plus',start='left',end='left',via=[(30,145),(30,1240)],label='Identity Q',label_at=(112,1150))
    d.edge('plus','out',start='right',end='left')
    d.bridge(320,215)
    d.footer(['Q_mix = (1 − G) ⊙ Q + G ⊙ Fw. The code uses the equivalent difference G ⊙ (Fw − Q).',
              'Q_out = Q + λ_gate × LN(Q_mix − Q). LayerNorm is inside the residual branch, not after the addition.',
              'Each decoder layer learns an independent λ_gate. If λ_gate = 0, Q_out equals Q exactly.'])
    d.save()


def overview():
    names=['01_RWA','02_GSQWCA','03_CQWG']
    panels=[]
    for name in names:
        with Image.open(HERE/f'{name}.png') as source:
            width=1200
            panels.append(source.convert('RGB').resize((width,round(source.height*width/source.width)),Image.Resampling.LANCZOS))
    height=max(panel.height for panel in panels)+150
    result=Image.new('RGB',(3660,height),'white')
    draw=ImageDraw.Draw(result)
    font=ImageFont.truetype(str(FONT_FILE),42) if FONT_FILE.exists() else ImageFont.load_default()
    draw.text((50,30),'Token-native multimodal modules: RWA → GSQWCA → CQWG',font=font,fill=INK)
    for index,panel in enumerate(panels): result.paste(panel,(index*1220+10,125))
    result.save(HERE/'water_modules_overview.png')


def main():
    HERE.mkdir(parents=True,exist_ok=True)
    draw_rwa(); draw_gsqwca(); draw_cqwg(); overview()
    manifest={'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'modules':['RankAwareWaterAttention','GlobalSparseQueryWaterCrossAttention','ComplementaryQueryWaterGate'],
              'formats':['png','svg','pdf'],'note':'Schematics match forward computation; diagnostic caches are not shown as computation branches.'}
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Saved three diagrams and overview to {HERE}')


if __name__=='__main__':
    main()
