# iOS

primeiro eu executei o build direto no xcode depois executei isso:

Build para iPhone (device)

```sh
xcodebuild archive \
  -project openlprsdk/openlprsdk.xcodeproj \
  -scheme openlprsdk \
  -destination "generic/platform=iOS" \
  -archivePath build/ios_devices.xcarchive \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES
```

Build para Simulator

```sh
xcodebuild archive \
  -project openlprsdk/openlprsdk.xcodeproj \
  -scheme openlprsdk \
  -destination "generic/platform=iOS Simulator" \
  -archivePath build/ios_simulator.xcarchive \
  SKIP_INSTALL=NO \
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES
```

Criar o XCFramework

```sh
xcodebuild -create-xcframework \
  -framework build/ios_devices.xcarchive/Products/Library/Frameworks/openlprsdk.framework \
  -framework build/ios_simulator.xcarchive/Products/Library/Frameworks/openlprsdk.framework \
  -output build/OpenLprSDK.xcframework
```

Compactar o resultado:

```sh
cd build
zip -r OpenLprSDK.xcframework.zip OpenLprSDK.xcframework
```

Dentro da pasta build que surgiu, eu tive que criar um arquivo chamado Package.swift com esse conteudo:

```swift
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "OpenLprSDK",
    platforms: [
        .iOS(.v13)
    ],
    products: [
        .library(
            name: "OpenLprSDK",
            targets: ["OpenLprSDK"]
        )
    ],
    targets: [
        .binaryTarget(
            name: "OpenLprSDK",
            path: "./OpenLprSDK.xcframework"
        )
    ]
)
```

ai eu consegui importar essa pasta lá no xcode como se fosse um pacote no meu projeto
