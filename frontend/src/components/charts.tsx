import {
  Bar, BarChart, CartesianGrid, ComposedChart, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis, Legend,
} from "recharts";
import type { Rep } from "../lib/types";

export const C = {
  accent: "#ff5a1f",
  ink: "#e9eaec",
  muted: "#8b9099",
  grid: "#24272c",
  blue: "#7aa7c7",
  sand: "#c9b48a",
};

const axis = { stroke: C.muted, fontSize: 11, tickLine: false, axisLine: { stroke: C.grid } };
const tip = {
  contentStyle: { background: "#1b1d21", border: "1px solid #30343a", borderRadius: 4, fontSize: 12 },
  labelStyle: { color: C.muted },
  itemStyle: { padding: 0 },
  cursor: { fill: "rgba(255,255,255,0.04)", stroke: "#3a3e45" },
};

export function RepChart({ reps, onset }: { reps: Rep[]; onset: number | null }) {
  const data = reps.map((r) => ({ rep: r.i, score: r.score, model: r.model_score, con: r.con_s, ecc: r.ecc_s, set: r.set }));
  const setStarts = reps.filter((r, k) => k > 0 && r.set !== reps[k - 1].set).map((r) => r.i);
  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={data} margin={{ top: 12, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="rep" {...axis} />
        <YAxis yAxisId="s" domain={[0, 100]} {...axis} width={40} />
        <YAxis yAxisId="t" orientation="right" {...axis} width={36} unit="s" />
        <Tooltip {...tip} />
        <Legend wrapperStyle={{ fontSize: 12, color: C.muted }} iconSize={10} />
        {setStarts.map((x) => <ReferenceLine key={x} yAxisId="s" x={x - 0.5} stroke="#3a3e45" strokeDasharray="2 3" />)}
        {onset && (
          <ReferenceLine yAxisId="s" x={onset} stroke={C.accent} strokeDasharray="4 3"
            label={{ value: "fatigue", fill: C.accent, fontSize: 11, position: "insideTopRight" }} />
        )}
        <Bar yAxisId="t" dataKey="con" name="Lift time (s)" fill="#3a4450" barSize={10} />
        <Line yAxisId="s" dataKey="score" name="Rule score" stroke={C.accent} strokeWidth={2} dot={{ r: 2.5 }} isAnimationActive={false} />
        <Line yAxisId="s" dataKey="model" name="Model % (synthetic-trained)" stroke={C.blue} strokeWidth={1.6} dot={false} strokeDasharray="5 3" isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function WeeklyBars({ data }: { data: { label: string; reps: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 4, bottom: 0, left: -16 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="label" {...axis} />
        <YAxis {...axis} width={44} allowDecimals={false} />
        <Tooltip {...tip} />
        <Bar dataKey="reps" name="Reps" fill={C.accent} radius={[2, 2, 0, 0]} maxBarSize={36} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function ScoreTrend({ data }: { data: { label: string; score: number; model: number | null }[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="label" {...axis} minTickGap={24} />
        <YAxis domain={[0, 100]} {...axis} width={44} />
        <Tooltip {...tip} />
        <Legend wrapperStyle={{ fontSize: 12 }} iconSize={10} />
        <Line dataKey="score" name="Rule score" stroke={C.accent} strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} connectNulls />
        <Line dataKey="model" name="Model % (synthetic-trained)" stroke={C.blue} strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} connectNulls />
      </LineChart>
    </ResponsiveContainer>
  );
}
