import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card } from "../ui/Card";

interface BarSeries<T> {
  key: keyof T;
  label: string;
  color?: string;
}

interface BarSummaryChartProps<T extends Record<string, any>> {
  title: string;
  description?: string;
  data: T[];
  xKey: keyof T;
  series: BarSeries<T>[];
}

export function BarSummaryChart<T extends Record<string, any>>({ title, description, data, xKey, series }: BarSummaryChartProps<T>) {
  return (
    <Card title={title} description={description} padded={false}>
      <div style={{ height: 320, padding: "0.5rem 1rem" }}>
        {data.length === 0 ? (
          <div style={{ padding: "1rem", color: "#475569" }}>No data to chart.</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 16, right: 24, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey={xKey as string} tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              {series.map((serie) => (
                <Bar key={String(serie.key)} dataKey={serie.key as string} name={serie.label} fill={serie.color || "#4f46e5"} radius={[6, 6, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </Card>
  );
}
