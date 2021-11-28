%--- Identification ------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cgMA}[2016/07/28 v1.9
                   Dokumentenklasse fuer Abschlussarbeiten
                   der AG Computergraphik]


%--- Initial code --------------------------------------------------------------

\RequirePackage{ifthen}
\newboolean{@isTimes}
\newboolean{@isPalatino}
\newboolean{@isExtern}


%--- Declaration of options ----------------------------------------------------

\DeclareOption{times}{\setboolean{@isTimes}{true}}
\DeclareOption{palatino}{\setboolean{@isPalatino}{true}}
\DeclareOption{twoside}{\PassOptionsToClass{twoside}{article}}
\DeclareOption{extern}{\setboolean{@isExtern}{true}}


%--- Execution of options ------------------------------------------------------

\ProcessOptions


%--- Package loading -----------------------------------------------------------

\LoadClass[a4paper,11pt,titlepage]{article}
\RequirePackage[english,ngerman]{babel}			% Unterstuetzung englische und neue deutsche Rechtschreibung
\RequirePackage[utf8]{inputenc}				% Unterstützung von Umlauten und ...
\RequirePackage[T1]{fontenc}					% Umlaute korrekt codieren
\RequirePackage{graphicx}					% fuer Einbindung von Grafiken
\RequirePackage{tabularx}					% flexible Gestaltung von Tabellen
\RequirePackage[hang,small,bf]{caption}			% Bildunterschriften mit Optionen
\RequirePackage{amsfonts}						%
\RequirePackage{amssymb}						%	Mathe-Zeugs
\RequirePackage{amsmath}						%
\RequirePackage{setspace} %Zeilenabstand ändern
        %       \singlespacing %einfach
        %       \onehalfspacing %1,5-fach
        %       \doublespacing %doppelt
        %       \linespread{1.25}
        %       \baselineskip15pt
\usepackage[margin=1.2in]{geometry}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%						% Default: Computer Modern (TeX-Standard)
\ifthenelse{\boolean{@isTimes}}{%
         \RequirePackage{times}					% Schriftart Times New Roman
      }{}
\ifthenelse{\boolean{@isPalatino}}{%
         \RequirePackage{palatino}			% Schriftart Book Antiqua/Palatino
      }{}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--- Code ----------------------------------------------------------------------

\def\@erstgutachter{}
\newcommand{\erstgutachter}[1]{\def\@erstgutachter{#1}}

\def\@erstgutachterInfo{}
\newcommand{\erstgutachterInfo}[1]{\def\@erstgutachterInfo{#1}}

\def\@zweitgutachter{}
\newcommand{\zweitgutachter}[1]{\def\@zweitgutachter{#1}}

\def\@zweitgutachterInfo{}
\newcommand{\zweitgutachterInfo}[1]{\def\@zweitgutachterInfo{#1}}

\def\@externLogo{}
\newcommand{\externLogo}[2]{\def\@externLogo{\includegraphics[width={#1}]{{#2}}}}

\def\@externName{}
\newcommand{\externName}[1]{\def\@externName{#1}}


\newcommand{\abgabe}{%
  \ifcase\month\or
   January\or February\or March\or April\or May\or June\or
   July\or August\or September\or October\or November\or December
   \fi\ \number\year
}


%%% save original offsets
\newlength{\voffsetBackup}
\setlength{\voffsetBackup}{\voffset}

\newlength{\hoffsetBackup}
\setlength{\hoffsetBackup}{\hoffset}

%%% preserve backslash (conflict with using \raggedleft
\newcommand{\PBS}[1]{\let\temp=\\#1\let\\=\temp}


%%% new titlepage
\renewcommand{\maketitle}{

   %%%  Titelseite  %%%
   \setlength{\voffset}{-.5in}
   \setlength{\hoffset}{-.25in}
   \begin{titlepage}
      \ifthenelse{\boolean{@twoside}}{%
         \setcounter{page}{0}
      }{%
         \setcounter{page}{-1}
      }
      % Logo Uni
      \hspace*{-1.7cm}
      % mit externem Logo
      \ifthenelse{\boolean{@isExtern}}{%
      	\begin{tabular*}{\linewidth}{p{.9cm}p{5cm}@{\hspace{2cm}}>{\PBS\raggedleft}p{6cm}}
 	     		{\includegraphics[height=2.14cm]{logos/uni}}%
 	     	&%
      		%\includegraphics[height=1.2cm]{logos/twoLogos}%
      	&%
     			\@externLogo%
	      \\[-.1ex] %
	      &%

      	&%
      		\footnotesize \@externName%
      \end{tabular*}
      }{%
      % nur Uni-Logo
      	\begin{tabular*}{\linewidth}{p{1.5cm}p{7cm}}
 	     		\raisebox{-.2cm}[1.3cm][1.3cm]{\includegraphics[height=2.14cm]{logos/twoLogos}}%
 	     	&%
      		%\includegraphics[height=1.3cm]{logos/twoLogos}%
	      \\[-1cm] %
	      &%
        \end{tabular*}
      }
      \normalsize
      \vspace*{1.5cm}
      \begin{center}
      \doublespacing
      {\huge\bfseries\@title}\\[1.25cm]
      \singlespacing
     	\huge Master's Thesis\\[.75cm]
     	\large In cooperation with: \\
     	\large wizAI solutions GmbH \\[.75cm]
      \normalsize in partial fulfillment of the requirements for\\the degree of Master
      of Science (M.Sc.)\\in Web and Data Science\\[.75cm]
      {\large submitted by\\[1ex]\Large\@author}\\[4cm]
			\end{center}
			\normalsize
      {\begin{tabular}{ll}
  	     First supervisor:   & \ifthenelse{\equal{\@erstgutachter}{}}{%
            {Prof. Dr.-Ing. Stefan M{\"u}ller}
            }{\@erstgutachter} \\
  	     { } & \ifthenelse{\equal{\@erstgutachterInfo}{}}{%
  	     		{(Institut f{\"u}r Computervisualistik, AG Computergraphik)}
  	     		}{\@erstgutachterInfo} \\[.2cm]
    	   Second supervisor: & \@zweitgutachter \\
    	   { } & \@zweitgutachterInfo \\
       \end{tabular}\\[1.5cm]}
  		\normalsize
  		\hspace*{.11cm}
      Koblenz, November 2021
   \end{titlepage}

   % Rueckseite der Titelseite
   \ifthenelse{\boolean{@twoside}}{%
      \begin{titlepage}
         \setcounter{page}{0}
         \mbox{}   % Hier muss was stehen, also eine leere Box :-)
      \end{titlepage}
   }{}

   \setlength{\voffset}{\voffsetBackup}
   \setlength{\hoffset}{\hoffsetBackup}


	%%%  Erklaerung  %%%
	\pagenumbering{gobble}		% remove page number
	\noindent
	\Large{Statement}\\

	\vspace*{2ex}
	\noindent
	\normalsize
	I hereby certify that this thesis has been composed by me and is based on my own
  work, that I did not use any further resources than specified --- in particular no
  references unmentioned in the reference section --- and that I did not submit this
  thesis to another examination before.\\

	\vspace*{2ex}

	\begin{flushleft}
	\begin{center}
	\begin{tabular}{l p{1cm} cc}
	\empty & \empty & Yes & No \\[.5cm]
	I agree to have this thesis published in the library.& \empty & $\square$ & $\boxtimes$\\[.5cm]
%   	I agree to have this thesis published on the Web.& \empty & $\square$ & $\boxtimes$\\[.5cm]
%   	The thesis text is available under a Creative Commons  & \empty & \empty & \empty\\
%                 License (CC BY-SA 4.0). & \empty & $\boxtimes$ & $\square$\\[.5cm]
%   	The source code is available under a GNU General Public & \empty & \empty & \empty\\
%                 License (GPLv3). & \empty & $\boxtimes$ & $\square$\\[.5cm]
%   	The collected data is available under a Creative Commons & \empty & \empty & \empty\\
%                 License (CC BY-SA 4.0). & \empty & $\square$ & $\boxtimes$ \\[.5cm]
	\end{tabular}\\[2cm]
	\end{center}
	\end{flushleft}
	\vspace*{3cm}

	\small
	\noindent
	\dotfill \\
	{\small(Place, Date)} \hfill{\small (Signature)}

	\normalsize

}

% count figures & tables per section
\numberwithin{figure}{section}
\renewcommand{\thefigure}{\arabic{section}.\arabic{figure}}
%\numberwithin{table}{section}
%\renewcommand{\thefigure}{\arabic{section}.\arabic{table}}