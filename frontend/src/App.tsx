import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/ui/AppShell";
import { LandingPage } from "./pages/LandingPage";
import { InputsResultsPage } from "./pages/InputsResultsPage";
import { SweepPage } from "./pages/SweepPage";
import { BatchPage } from "./pages/BatchPage";

const routes = [
  { path: "/", element: <LandingPage /> },
  { path: "/uploads", element: <LandingPage /> },
  { path: "/inputs", element: <InputsResultsPage /> },
  { path: "/sweep", element: <SweepPage /> },
  { path: "/batch", element: <BatchPage /> },
];

function App() {
  return (
    <AppShell>
      <Routes>
        {routes.map((route) => (
          <Route key={route.path} path={route.path} element={route.element} />
        ))}
      </Routes>
    </AppShell>
  );
}

export default App;
