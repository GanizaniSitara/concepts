export default `
core {
	active-bg-color: #fff;
	active-bg-opacity: 0.333;
}

edge {
	curve-style: haystack;
	haystack-radius: 0;
	opacity: 0.333;
	width: 2;
	z-index: 0;
	overlay-opacity: 0;
    events: no;
}

edge[interaction = "Depends on::Used by"] {
	line-color: #FACD37;
	opacity: 0.666;
	z-index: 9;
	width: 4;
	curve-style: bezier;
	target-arrow-shape: triangle;
	target-arrow-color: #FFFFFF;
}

edge[interaction = "cr"] {
	line-color: #DE3128;
}

edge[interaction = "cw"] {
	line-color: white;
}

edge[arrow]{
   target-arrow-shape: triangle;
}

node {
	shape: round-rectangle;
	width: 80;
	height: 40;
	font-size: 9;
	font-weight: bold;
	min-zoomed-font-size: 4;
	label: data(name);
	text-wrap: wrap;
	text-max-width: 50;
	text-valign: center;
	text-halign: center;
	text-events: yes;
	color: #000;
	text-outline-width: 1;
	text-outline-color: #fff;
	text-outline-opacity: 1;
	overlay-color: #fff;
}

node[Degree >= 50] {
    height: 350;
    width: 350;
    font-size: 48;
}

node[Owning_Transaction_Cycle = 'Retail Banking Platform']
  {
    background-color : rgb(86,166,218);
    text-outline-color: rgb(86,166,218);
  }

node[Owning_Transaction_Cycle = 'Cards and Payments']
  {
    background-color : rgb(255,0,102);
    text-outline-color: rgb(255,0,102);
  }

node[Owning_Transaction_Cycle = 'Mortgage Services']
  {
    background-color : rgb(218,226,66);
    text-outline-color: rgb(218,226,66);
  }

node[Owning_Transaction_Cycle = 'Wealth Management']
  {
    background-color : rgb(153,102,0);
    text-outline-color: rgb(153,102,0);
  }

node[Owning_Transaction_Cycle = 'Private Banking Operations']
  {
    background-color : rgb(153,255,204);
    text-outline-color: rgb(153,255,204);
  }

node[Owning_Transaction_Cycle = 'Markets Pre-Trade']
  {
    background-color : rgb(204,204,0);
    text-outline-color: rgb(204,204,0);
  }

node[Owning_Transaction_Cycle = 'Markets Post-Trade']
  {
    background-color : rgb(255,102,255);
    text-outline-color: rgb(255,102,255);
  }

node[Owning_Transaction_Cycle = 'Corporate Lending']
  {
    background-color : rgb(0,204,0);
    text-outline-color: rgb(0,204,0);
  }

node[Owning_Transaction_Cycle = 'Trade Finance']
  {
    background-color : rgb(0,255,153);
    text-outline-color: rgb(0,255,153);
  }

node[Owning_Transaction_Cycle = 'Transaction Banking']
  {
    background-color : rgb(204,255,102);
    text-outline-color: rgb(204,255,102);
  }

node[Owning_Transaction_Cycle = 'Global Payments']
  {
    background-color : rgb(102,102,255);
    text-outline-color: rgb(102,102,255);
  }

node[Owning_Transaction_Cycle = 'Merchant Acquiring']
  {
    background-color : rgb(204,204,255);
    text-outline-color: rgb(204,204,255);
  }

node[Owning_Transaction_Cycle = 'Risk and Compliance']
  {
    background-color : rgb(153,153,255);
    text-outline-color: rgb(153,153,255);
  }

node[Owning_Transaction_Cycle = 'Finance and Treasury']
  {
    background-color : rgb(255,153,204);
    text-outline-color: rgb(255,153,204);
  }

node[Owning_Transaction_Cycle = 'Chief Data Office']
  {
    background-color : rgb(65,182,196);
    text-outline-color: rgb(65,182,196);
  }

node[Owning_Transaction_Cycle = 'Enterprise Technology']
  {
    background-color : rgb(255,153,0);
    text-outline-color: rgb(255,153,0);
  }

node[Owning_Transaction_Cycle = 'Security Operations']
  {
    background-color : rgb(241,105,19);
    text-outline-color: rgb(241,105,19);
  }

node[Owning_Transaction_Cycle = 'Customer Digital']
  {
    background-color : rgb(0,153,153);
    text-outline-color: rgb(0,153,153);
  }

node[Owning_Transaction_Cycle = 'Digital Channels']
  {
    background-color : rgb(153,255,255);
    text-outline-color: rgb(153,255,255);
  }

node[Owning_Transaction_Cycle = 'Infrastructure Services']
  {
    background-color : rgb(191,211,230);
    text-outline-color: rgb(191,211,230);
  }

node[Owning_Transaction_Cycle = 'Service Management']
  {
    background-color : rgb(255,204,204);
    text-outline-color: rgb(255,204,204);
  }

node[Owning_Transaction_Cycle = 'HR and Corporate Services']
  {
    background-color : rgb(35,132,67);
    text-outline-color: rgb(35,132,67);
  }

node[Owning_Transaction_Cycle = 'Insurance Platform']
  {
    background-color : rgb(255,153,153);
    text-outline-color: rgb(255,153,153);
  }

node[Owning_Transaction_Cycle = 'Client Onboarding']
  {
    background-color : rgb(0,204,204);
    text-outline-color: rgb(0,204,204);
  }

node[Owning_Transaction_Cycle = 'Regulatory Reporting']
  {
    background-color : rgb(0,102,153);
    text-outline-color: rgb(0,102,153);
  }

node.highlighted {
	min-zoomed-font-size: 0;
  z-index: 9999;
}

edge.highlighted {
	opacity: 0.8;
	width: 4;
	z-index: 9999;
}

.faded {
  events: no;
}

node.faded {
  opacity: 0.08;
}

edge.faded {
  opacity: 0.06;
}

.hidden {
	display: none;
}

`;
