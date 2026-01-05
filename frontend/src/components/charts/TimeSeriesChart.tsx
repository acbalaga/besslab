import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Card } from "../ui/Card";

interface Series<T> {
  key: keyof T;
  label: string;
  color?: string;
}

interface TimeSeriesChartProps<T extends Record<string, number | string>> {
  title: string;
  description?: string;
  data: T[];
  xKey: keyof T;
  series: Series<T>[];
  unit?: string;
}

export function TimeSeriesChart<T extends Record<string, number | string>>({ title, description, data, xKey, series, unit }: TimeSeriesChartProps<T>) {
  return (
    <Card title={title} description={description} padded={false}>
      <div style={{ height: 320, padding: "0.5rem 1rem" }}>
        {data.length === 0 ? (
          <div style={{ padding: "1rem", color: "#475569" }}>No chart data available.</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 16, right: 24, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey={xKey as string} tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} unit={unit} />
              <Tooltip />
              <Legend />
              {series.map((serie) => (
                <Line
                  key={String(serie.key)}
                  type="monotone"
                  dataKey={serie.key as string}
                  name={serie.label}
                  stroke={serie.color || "#4f46e5"}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </Card>
  );
}
