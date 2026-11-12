export default function Gauge({ value }) {
  const ratio = Math.min(1, Math.max(0, value));
  const start = Math.PI;
  const end = Math.PI * (1 - ratio);
  const radius = 86;
  const cx = 120;
  const cy = 108;

  const point = (angle) => [cx + radius * Math.cos(angle), cy - radius * Math.sin(angle)];
  const arc = (a0, a1) => {
    const [x0, y0] = point(a0);
    const [x1, y1] = point(a1);
    const large = a0 - a1 > Math.PI ? 1 : 0;
    return `M ${x0} ${y0} A ${radius} ${radius} 0 ${large} 1 ${x1} ${y1}`;
  };
  const [nx, ny] = point(end);

  return (
    <svg className="gauge" viewBox="0 0 240 132" role="img" aria-label="Jauge de probabilité de défaut">
      <path d={arc(start, 0)} className="gauge-track" />
      <path d={arc(start, end)} className="gauge-value" />
      <circle cx={nx} cy={ny} r="5" className="gauge-dot" />
      <text x="120" y="96" textAnchor="middle" className="gauge-figure">
        {Math.round(ratio * 1000) / 10}
        <tspan className="gauge-unit"> %</tspan>
      </text>
      <text x="28" y="126" className="gauge-bound">
        0
      </text>
      <text x="206" y="126" className="gauge-bound">
        100
      </text>
    </svg>
  );
}
